#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import telnetlib
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional, Sequence, Tuple

import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)

JOINT_NAMES = ("J1", "J2", "J3", "J4")
JOINT_LIMITS = {
    "J1": (0.0015, 1.0),
    "J2": (-1.62316, 1.62316),
    "J3": (0.20944, 6.07375),
    "J4": (-16.7552, 16.7552),
}
DEFAULT_MAX_DELTA = (0.01, 0.06, 0.06, 0.09)


class CameraError(RuntimeError):
    pass


class TCSCommandError(RuntimeError):
    def __init__(self, command: str, reply: str) -> None:
        super().__init__(f"TCS error on {command!r}: {reply}")
        self.command = command
        self.reply = reply


class JointSafetyError(ValueError):
    pass


class RealSenseSource:
    """Minimal RGB RealSense source matching the SCARA recording setup."""

    def __init__(
        self,
        color_name: str,
        size: Tuple[int, int] = (640, 480),
        fps: int = 30,
        serial: Optional[str] = None,
        warmup_frames: int = 30,
        timeout_ms: int = 1000,
    ) -> None:
        self.color_name = color_name
        self.size = tuple(size)
        self.fps = int(fps)
        self.serial = serial
        self.warmup_frames = int(warmup_frames)
        self.timeout_ms = int(timeout_ms)
        self.rs: Any = None
        self.pipeline: Any = None

    def start(self) -> "RealSenseSource":
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError("pyrealsense2 is required for live SCARA evaluation") from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        width, height = self.size
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, self.fps)
        try:
            self.pipeline.start(config)
        except Exception as exc:
            self.pipeline = None
            raise CameraError(f"Could not start RealSense camera: {exc}") from exc

        for _ in range(self.warmup_frames):
            self._wait_for_frames()
        return self

    def _wait_for_frames(self):
        if self.pipeline is None:
            raise CameraError("Camera has not been started")
        try:
            return self.pipeline.wait_for_frames(self.timeout_ms)
        except Exception as exc:
            raise CameraError("Timed out waiting for a RealSense frame") from exc

    def read(self) -> dict:
        frameset = self._wait_for_frames()
        color_frame = frameset.get_color_frame()
        if not color_frame:
            raise CameraError("RealSense returned no RGB frame")
        return {
            "image": np.asanyarray(color_frame.get_data()).copy(),
            "timestamp": time.monotonic(),
            "rs_timestamp_ms": float(color_frame.get_timestamp()),
            "frame_number": int(color_frame.get_frame_number()),
        }

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


class TCSBaseClient:
    def __init__(
        self,
        host: str = "192.168.0.10",
        port: int = 10100,
        timeout: float = 5.0,
        verbose: bool = False,
    ) -> None:
        self.verbose = bool(verbose)
        self.lock = threading.Lock()
        self.connection = telnetlib.Telnet(host, int(port), float(timeout))

    def command(self, command: str) -> str:
        if self.verbose:
            print(f"TCS command: {command}", flush=True)
        with self.lock:
            self.connection.write((command + "\n").encode("ascii"))
            line = self.connection.read_until(b"\n").decode("ascii").strip()
        if not line:
            raise TCSCommandError(command, "empty reply")
        parts = line.split(" ", 1)
        if parts[0].startswith("-"):
            raise TCSCommandError(command, line)
        return parts[1] if len(parts) > 1 else ""

    def command_sleep(self, command: str, delay: float = 0.15) -> str:
        reply = self.command(command)
        time.sleep(max(0.0, float(delay)))
        return reply

    def close(self) -> None:
        self.connection.close()


class TCSReadClient(TCSBaseClient):
    def get_joint_vector(self) -> np.ndarray:
        values = [float(value) for value in self.command("wherej").split()]
        if len(values) < len(JOINT_NAMES):
            raise ValueError(f"Expected four values from wherej, got {values!r}")
        return np.asarray(
            [
                values[0] * 0.001,
                values[1] * math.pi / 180.0,
                values[2] * math.pi / 180.0,
                values[3] * math.pi / 180.0,
            ],
            dtype=np.float32,
        )


class TCSMotionClient(TCSReadClient):
    def __init__(
        self,
        host: str = "192.168.0.10",
        port: int = 10100,
        timeout: float = 5.0,
        verbose: bool = False,
        profile: int = 2,
        mspeed: int = 100,
        profile_speed: int = 20,
        profile_accel: int = 20,
        profile_ramp: float = 0.08,
        profile_straight: int = 0,
    ) -> None:
        super().__init__(host=host, port=port, timeout=timeout, verbose=verbose)
        self.profile = int(profile)
        try:
            self._startup()
            self._configure_motion(
                mspeed=mspeed,
                profile_speed=profile_speed,
                profile_accel=profile_accel,
                profile_ramp=profile_ramp,
                profile_straight=profile_straight,
            )
        except BaseException:
            self.close()
            raise

    def _startup(self) -> None:
        self.command_sleep("SelectRobot 1")
        self.command_sleep("attach 1")
        self.command_sleep("hp 1 -1")
        if self.command_sleep("attach") != "1":
            raise TCSCommandError("attach", "robot 1 is not attached")
        if self.command_sleep("hp") != "1":
            raise TCSCommandError("hp", "high power is not enabled")
        self.get_joint_vector()

    def _configure_motion(
        self,
        mspeed: int,
        profile_speed: int,
        profile_accel: int,
        profile_ramp: float,
        profile_straight: int,
    ) -> None:
        self.command(f"mspeed {int(mspeed)}")
        self.command(
            f"profile {self.profile} {int(profile_speed)} {int(profile_speed)} "
            f"{int(profile_accel)} {int(profile_accel)} "
            f"{float(profile_ramp)} {float(profile_ramp)} -1 {int(profile_straight)}"
        )

    def movej(self, target: Sequence[float]) -> None:
        target = require_safe_joint_target(target)
        values = (
            target[0] * 1000.0,
            target[1] * 180.0 / math.pi,
            target[2] * 180.0 / math.pi,
            target[3] * 180.0 / math.pi,
        )
        self.command(
            f"MoveJ {self.profile} " + " ".join(f"{value:.6g}" for value in values)
        )

    def halt(self) -> None:
        self.command("halt")


def require_safe_joint_target(target: Sequence[float]) -> np.ndarray:
    values = np.asarray(target, dtype=np.float32)
    if values.shape != (len(JOINT_NAMES),):
        raise JointSafetyError(f"Expected a four-joint target, got shape {values.shape}")
    violations = []
    for index, joint_name in enumerate(JOINT_NAMES):
        value = float(values[index])
        lower, upper = JOINT_LIMITS[joint_name]
        if not math.isfinite(value):
            violations.append(f"{joint_name} is not finite")
        elif not lower <= value <= upper:
            violations.append(
                f"{joint_name}={value:.6g} outside [{lower:.6g}, {upper:.6g}]"
            )
    if violations:
        raise JointSafetyError("; ".join(violations))
    return values


def validate_action_chunk(
    actions: np.ndarray,
    current_qpos: np.ndarray,
    max_delta: Sequence[float],
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(JOINT_NAMES):
        raise JointSafetyError(f"Expected action shape (T, 4), got {actions.shape}")
    maximum = np.asarray(max_delta, dtype=np.float32)
    if maximum.shape != (len(JOINT_NAMES),) or np.any(maximum <= 0):
        raise ValueError("max_delta must contain four positive values")

    previous = np.asarray(current_qpos, dtype=np.float32)
    checked = []
    for index, action in enumerate(actions):
        target = require_safe_joint_target(action)
        delta = np.abs(target - previous)
        if np.any(delta > maximum):
            raise JointSafetyError(
                f"Action {index} delta {delta.tolist()} exceeds {maximum.tolist()}"
            )
        checked.append(target)
        previous = target
    return np.stack(checked)


def load_policy(
    checkpoint: Path,
    device: torch.device,
    num_inference_steps: Optional[int],
) -> Tuple[BaseImagePolicy, OmegaConf]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with checkpoint.open("rb") as source:
        payload = torch.load(
            source,
            pickle_module=dill,
            map_location="cpu",
        )
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_cls(cfg, output_dir=str(checkpoint.parent))
    workspace.load_payload(payload)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    policy.reset()
    if num_inference_steps is not None:
        if not hasattr(policy, "num_inference_steps"):
            raise ValueError("This policy has no configurable inference step count")
        policy.num_inference_steps = int(num_inference_steps)
    return policy, cfg


def get_scara_shape_meta(cfg: OmegaConf) -> Tuple[str, Tuple[int, int, int]]:
    rgb_keys = [
        str(key)
        for key, attributes in cfg.task.shape_meta.obs.items()
        if str(attributes.get("type", "low_dim")) == "rgb"
    ]
    lowdim_keys = [
        str(key)
        for key, attributes in cfg.task.shape_meta.obs.items()
        if str(attributes.get("type", "low_dim")) == "low_dim"
    ]
    if len(rgb_keys) != 1 or lowdim_keys != ["qpos"]:
        raise ValueError(
            f"Expected one RGB key and qpos, got RGB={rgb_keys}, low_dim={lowdim_keys}"
        )
    image_shape = tuple(int(value) for value in cfg.task.shape_meta.obs[rgb_keys[0]].shape)
    action_shape = tuple(int(value) for value in cfg.task.shape_meta.action.shape)
    if len(image_shape) != 3 or image_shape[0] != 3:
        raise ValueError(f"Expected RGB shape (3, H, W), got {image_shape}")
    if action_shape != (len(JOINT_NAMES),):
        raise ValueError(f"Expected four-dimensional actions, got {action_shape}")
    return rgb_keys[0], image_shape


def image_to_chw(image: np.ndarray, image_shape: Tuple[int, int, int]) -> np.ndarray:
    channels, target_height, target_width = image_shape
    if image.ndim != 3 or image.shape[-1] != channels:
        raise ValueError(f"Expected an HWC RGB image, got {image.shape}")
    if image.shape[:2] != (target_height, target_width):
        shrinking = image.shape[0] > target_height or image.shape[1] > target_width
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        image = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=interpolation,
        )
    return np.ascontiguousarray(np.moveaxis(image, -1, 0), dtype=np.float32) / 255.0


def make_policy_observation(
    history: Deque[Tuple[np.ndarray, np.ndarray]],
    camera_key: str,
    image_shape: Tuple[int, int, int],
    device: torch.device,
) -> dict:
    if not history:
        raise ValueError("Observation history is empty")
    images = np.stack([image_to_chw(item[0], image_shape) for item in history])
    qpos = np.stack([item[1] for item in history]).astype(np.float32)
    return {
        camera_key: torch.from_numpy(images).unsqueeze(0).to(device),
        "qpos": torch.from_numpy(qpos).unsqueeze(0).to(device),
    }


def capture_history(
    camera: RealSenseSource,
    robot: TCSReadClient,
    n_obs_steps: int,
) -> Deque[Tuple[np.ndarray, np.ndarray]]:
    history: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=n_obs_steps)
    for _ in range(n_obs_steps):
        sample = camera.read()
        history.append((sample["image"], robot.get_joint_vector()))
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained Stanford Diffusion Policy on the SCARA setup."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--start-action", type=int, default=0)
    parser.add_argument("--action-count", type=int, default=1)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--show-camera", action="store_true")
    parser.add_argument("--host", default="192.168.0.10")
    parser.add_argument("--port", type=int, default=10100)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--serial", default="130322273198")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--max-delta", type=float, nargs=4, default=DEFAULT_MAX_DELTA)
    parser.add_argument("--profile", type=int, default=2)
    parser.add_argument("--mspeed", type=int, default=100)
    parser.add_argument("--profile-speed", type=int, default=20)
    parser.add_argument("--profile-accel", type=int, default=20)
    parser.add_argument("--profile-ramp", type=float, default=0.08)
    parser.add_argument("--profile-straight", type=int, default=0)
    parser.add_argument("--verbose-tcs", action="store_true")
    args = parser.parse_args()
    if args.chunks < 1 or args.action_count < 1 or args.hz <= 0:
        parser.error("chunks, action-count and hz must be positive")
    if args.start_action < 0:
        parser.error("start-action cannot be negative")
    if args.num_inference_steps < 1:
        parser.error("num-inference-steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    policy, cfg = load_policy(
        checkpoint=args.checkpoint.expanduser().resolve(),
        device=device,
        num_inference_steps=args.num_inference_steps,
    )
    camera_key, image_shape = get_scara_shape_meta(cfg)
    n_obs_steps = int(cfg.n_obs_steps)
    camera = RealSenseSource(
        color_name=camera_key,
        size=(args.camera_width, args.camera_height),
        fps=args.camera_fps,
        serial=args.serial or None,
    )
    robot: Optional[TCSReadClient] = None

    try:
        camera.start()
        client_cls = TCSMotionClient if args.execute else TCSReadClient
        client_kwargs = {
            "host": args.host,
            "port": args.port,
            "timeout": args.timeout,
            "verbose": args.verbose_tcs,
        }
        if args.execute:
            client_kwargs.update(
                {
                    "profile": args.profile,
                    "mspeed": args.mspeed,
                    "profile_speed": args.profile_speed,
                    "profile_accel": args.profile_accel,
                    "profile_ramp": args.profile_ramp,
                    "profile_straight": args.profile_straight,
                }
            )
        robot = client_cls(**client_kwargs)
        history = capture_history(camera, robot, n_obs_steps)
        policy.reset()

        mode = "EXECUTE" if args.execute else "READ-ONLY"
        print(
            f"Mode={mode}, image={image_shape}, obs_steps={n_obs_steps}, "
            f"inference_steps={args.num_inference_steps}",
            flush=True,
        )
        if args.execute:
            print("Robot movement enabled; keep the emergency stop within reach.", flush=True)
            print(f"Joint limits: {JOINT_LIMITS}", flush=True)
            print(f"Maximum per-command delta: {tuple(args.max_delta)}", flush=True)
            for seconds in range(3, 0, -1):
                print(f"Starting policy control in {seconds}...", flush=True)
                time.sleep(1.0)

        for chunk_index in range(args.chunks):
            obs = make_policy_observation(history, camera_key, image_shape, device)
            inference_start = time.monotonic()
            with torch.inference_mode():
                prediction = policy.predict_action(obs)["action"][0].detach().cpu().numpy()
            inference_latency = time.monotonic() - inference_start

            selection_end = args.start_action + args.action_count
            if selection_end > len(prediction):
                raise ValueError(
                    f"Requested actions [{args.start_action}:{selection_end}], "
                    f"but policy returned {len(prediction)}"
                )
            selected = prediction[args.start_action:selection_end]
            current_qpos = robot.get_joint_vector()
            selected = validate_action_chunk(selected, current_qpos, args.max_delta)

            print(
                f"chunk {chunk_index + 1}/{args.chunks}, "
                f"inference={inference_latency * 1000.0:.1f} ms",
                flush=True,
            )
            print("current qpos:", np.round(current_qpos, 5), flush=True)
            print("selected actions:\n", np.round(selected, 5), flush=True)

            if args.show_camera:
                cv2.imshow("SCARA Diffusion Policy", history[-1][0][..., ::-1])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            period = 1.0 / args.hz
            for action in selected:
                command_start = time.monotonic()
                if args.execute:
                    assert isinstance(robot, TCSMotionClient)
                    robot.movej(action)
                time.sleep(max(0.0, period - (time.monotonic() - command_start)))

            history = capture_history(camera, robot, n_obs_steps)
    finally:
        if args.execute and isinstance(robot, TCSMotionClient):
            try:
                robot.halt()
                time.sleep(0.3)
                robot.command("nop")
                robot.command("attach 0")
            except Exception as exc:
                print(f"Warning during TCS shutdown: {exc}", file=sys.stderr)
        if robot is not None:
            robot.close()
        camera.close()
        if args.show_camera:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
