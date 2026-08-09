#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scara_diffusion_policy.constants import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_MAX_DELTA,
)
from scara_diffusion_policy.dataset import (  # noqa: E402
    NormalizationStats,
    fit_dim,
    image_to_chw,
)
from scara_diffusion_policy.policy import ScaraDiffusionPolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run a SCARA Diffusion Policy")
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--checkpoint", default="policy_best.ckpt")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--start-action", type=int, default=0)
    parser.add_argument("--action-count", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Camera + read-only robot state.")
    mode.add_argument("--execute", action="store_true", help="Send the predicted targets to TCS.")
    parser.add_argument("--host", default="192.168.0.10")
    parser.add_argument("--port", type=int, default=10100)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-delta", type=float, nargs=4, default=DEFAULT_MAX_DELTA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-camera-view", action="store_true")
    return parser.parse_args()


def load_policy(ckpt_dir: Path, checkpoint: str, device: torch.device):
    config = json.loads((ckpt_dir / "train_config.json").read_text())
    if config.get("format_version") != 2:
        raise RuntimeError("This is an old z-score checkpoint; train a new v2 policy first")
    with (ckpt_dir / "dataset_stats.pkl").open("rb") as source:
        stats = NormalizationStats.from_dict(pickle.load(source))
    model_settings = dict(config["model"])
    model_settings["pretrained_backbone"] = False
    policy = ScaraDiffusionPolicy(**model_settings)
    policy.set_normalizer(stats)
    policy.load_state_dict(
        torch.load(ckpt_dir / checkpoint, map_location="cpu", weights_only=True)
    )
    return policy.to(device).eval(), config


def image_dataset(root: h5py.File, camera_name: str):
    rgb = f"observations/images/{camera_name}"
    depth = f"observations/depth_images/{camera_name}"
    if rgb in root:
        return root[rgb]
    if depth in root:
        return root[depth]
    raise KeyError(f"camera {camera_name!r} not found")


def dataset_input(config: dict, dataset_dir: str, episode: int, frame: int):
    path = Path(dataset_dir) / f"episode_{episode}.hdf5"
    if not path.exists():
        raise FileNotFoundError(path)
    n_obs = int(config["model"]["n_obs_steps"])
    n_actions = int(config["model"]["n_action_steps"])
    with h5py.File(path, "r") as root:
        length = int(root["observations/qpos"].shape[0])
        if not 0 <= frame < length:
            raise IndexError(f"frame {frame} outside episode length {length}")
        obs_indices = np.clip(np.arange(frame - n_obs + 1, frame + 1), 0, length - 1)
        action_indices = np.clip(np.arange(frame, frame + n_actions), 0, length - 1)
        history = [
            (
                fit_dim(root["observations/qpos"][index], config["model"]["state_dim"]),
                {
                    name: np.asarray(image_dataset(root, name)[index])
                    for name in config["camera_names"]
                },
            )
            for index in obs_indices
        ]
        target = fit_dim(
            np.stack([root["action"][index] for index in action_indices]),
            config["model"]["state_dim"],
        )
    return history, target


def make_batch(history, config: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch = {
        "qpos": torch.from_numpy(np.stack([item[0] for item in history]))[None].to(device)
    }
    for name in config["camera_names"]:
        _, height, width = config["image_shapes"][name]
        images = np.stack(
            [image_to_chw(item[1][name], (width, height)) for item in history]
        )
        batch[name] = torch.from_numpy(images)[None].to(device)
    return batch


def predict(policy, history, config, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    start = time.monotonic()
    result = policy.predict_action(make_batch(history, config, device), generator=generator)
    return result["action"][0].cpu().numpy(), time.monotonic() - start


def live_input(camera, robot):
    sample = camera.read()
    from scara_diffusion_policy.tcs_client import joint_dict_to_vector

    return joint_dict_to_vector(robot.get_joint_state()), sample["frames"]


def live_history(camera, robot, count):
    return [live_input(camera, robot) for _ in range(count)]


def main() -> None:
    args = parse_args()
    if args.chunks < 1 or args.hz <= 0:
        raise ValueError("chunks and Hz must be positive")

    device = torch.device(args.device)
    policy, config = load_policy(Path(args.ckpt_dir), args.checkpoint, device)
    available = int(config["model"]["n_action_steps"])
    count = args.action_count if args.action_count is not None else available - args.start_action
    if args.start_action < 0 or count < 1 or args.start_action + count > available:
        raise ValueError(f"choose actions inside the returned chunk of {available}")
    selected = slice(args.start_action, args.start_action + count)
    np.set_printoptions(precision=5, suppress=True)

    if not args.live and not args.execute:
        for chunk in range(args.chunks):
            frame = args.frame + chunk * available
            history, target = dataset_input(
                config, args.dataset_dir, args.episode, frame
            )
            actions, latency = predict(
                policy, history, config, device, args.seed + chunk
            )
            actions, target = actions[selected], target[selected]
            print(f"chunk {chunk}, episode {args.episode}, frame {frame}, {latency*1000:.1f} ms")
            print("predicted:\n", actions)
            print("recorded:\n", target)
            print("MAE per joint:", np.abs(actions - target).mean(0))
        return

    from scara_diffusion_policy.camera import RealSenseSource
    from scara_diffusion_policy.tcs_client import (
        TCSMotionClient,
        TCSReadClient,
        joint_dict_to_vector,
        require_joint_target_within_limits,
        vector_to_joint_dict,
    )

    camera_names = config["camera_names"]
    color_names = [name for name in camera_names if not name.endswith("_depth")]
    depth_names = [name for name in camera_names if name.endswith("_depth")]
    if len(color_names) != 1 or len(depth_names) > 1:
        raise ValueError("live eval supports one RGB camera and at most one depth stream")
    camera = RealSenseSource(
        color_name=color_names[0],
        depth_name=depth_names[0] if depth_names else None,
        serial=args.serial,
        fps=30,
        size=(640, 480),
        align_depth=bool(depth_names),
    )
    robot = None
    viewer = None
    try:
        camera.start()
        if not args.no_viewer:
            from scara_diffusion_policy.live_viewer import LiveViewer

            viewer = LiveViewer(show_camera=not args.no_camera_view)
        client = TCSMotionClient if args.execute else TCSReadClient
        robot = client(host=args.host, port=args.port)
        n_obs = int(config["model"]["n_obs_steps"])
        history = live_history(camera, robot, n_obs)

        for chunk in range(args.chunks):
            actions, latency = predict(
                policy, history, config, device, args.seed + chunk
            )
            actions = actions[selected]
            print(f"chunk {chunk}, inference {latency*1000:.1f} ms\n{actions}", flush=True)
            if viewer is not None and not viewer.update(
                history[-1][1][color_names[0]],
                history[-1][0],
                actions,
                chunk + 1,
            ):
                print("Viewer closed; stopping.", flush=True)
                break

            targets = []
            if args.execute:
                previous = joint_dict_to_vector(robot.get_joint_state())
                for action in actions:
                    target = require_joint_target_within_limits(vector_to_joint_dict(action))
                    delta = np.abs(action - previous)
                    if np.any(delta > np.asarray(args.max_delta)):
                        raise RuntimeError(
                            f"target delta {delta} exceeds max {np.asarray(args.max_delta)}"
                        )
                    targets.append(target)
                    previous = action

            period = 1.0 / args.hz
            for index in range(len(actions)):
                command_start = time.monotonic()
                if args.execute:
                    robot.movej(targets[index])
                time.sleep(max(0.0, period - (time.monotonic() - command_start)))
            history = live_history(camera, robot, n_obs)
    finally:
        if args.execute and robot is not None:
            try:
                robot.halt()
                time.sleep(0.3)
                robot.command("nop")
                robot.command("attach 0")
            except Exception as exc:
                print(f"Warning during TCS shutdown: {exc}", file=sys.stderr)
        if viewer is not None:
            viewer.close()
        camera.close()
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    main()
