#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scara_diffusion_policy.constants import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_DIR,
    JOINT_LIMITS,
    JOINT_NAMES,
    STATE_DIM,
    UPSTREAM_DIR,
)
from scara_diffusion_policy.dataset import (  # noqa: E402
    NormalizationStats,
    ScaraDiffusionDataset,
    calculate_stats,
    episode_id,
    list_episodes,
)
from scara_diffusion_policy.policy import ScaraDiffusionPolicy  # noqa: E402


if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from diffusion_policy.model.common.lr_scheduler import get_scheduler  # noqa: E402
from diffusion_policy.model.diffusion.ema_model import EMAModel  # noqa: E402


FORMAT_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train the SCARA image Diffusion Policy")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--camera-names", default=",".join(CAMERA_NAMES))
    parser.add_argument("--state-dim", type=int, default=STATE_DIM)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-batches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-val-batches", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be between zero and one")
    if args.state_dim != STATE_DIM:
        parser.error(f"this SCARA port requires --state-dim {STATE_DIM}")
    if min(args.horizon, args.n_obs_steps, args.n_action_steps, args.num_inference_steps) < 1:
        parser.error("horizon, observation, action and inference steps must be positive")
    if args.n_action_steps > args.horizon - args.n_obs_steps + 1:
        parser.error("--n-action-steps does not fit in the prediction horizon")
    if args.num_epochs is not None and args.num_epochs < 1:
        parser.error("--num-epochs must be positive")
    if min(args.batch_size, args.image_width, args.image_height) < 1:
        parser.error("batch size and image dimensions must be positive")
    if args.save_every < 1:
        parser.error("--save-every must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def mean_loss(
    model: ScaraDiffusionPolicy,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    use_amp: bool,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model.compute_loss(move_batch(batch, device), generator=generator)
            losses.append(float(loss.cpu()))
    if not losses:
        raise RuntimeError("validation loader produced no batches")
    return float(np.mean(losses))


def model_config(args: argparse.Namespace, camera_names: list[str]) -> dict:
    return {
        "camera_names": camera_names,
        "state_dim": args.state_dim,
        "horizon": args.horizon,
        "n_obs_steps": args.n_obs_steps,
        "n_action_steps": args.n_action_steps,
        "num_inference_steps": args.num_inference_steps,
        "down_dims": [256, 512, 1024],
        "pretrained_backbone": args.pretrained_backbone,
        "freeze_backbone": args.freeze_backbone,
    }


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_pickle_save(value, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as target:
        pickle.dump(value, target)
    temporary.replace(path)


def save_training_state(
    path: Path,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    model: ScaraDiffusionPolicy,
    ema_model: ScaraDiffusionPolicy | None,
    ema: EMAModel | None,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: dict,
) -> None:
    atomic_torch_save(
        {
            "format_version": FORMAT_VERSION,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "ema_optimization_step": ema.optimization_step if ema is not None else 0,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    ckpt_dir = Path(args.ckpt_dir)
    state_path = ckpt_dir / "training_state_last.pt"
    config_path = ckpt_dir / "train_config.json"
    stats_path = ckpt_dir / "dataset_stats.pkl"
    saved_config = json.loads(config_path.read_text()) if args.resume and config_path.exists() else None
    if args.resume and (saved_config is None or not state_path.exists() or not stats_path.exists()):
        raise FileNotFoundError(f"No complete resumable run found in {ckpt_dir}")
    if not args.resume and (state_path.exists() or config_path.exists()):
        raise FileExistsError(f"{ckpt_dir} already contains a run; use --resume or another directory")
    if saved_config is not None:
        if saved_config.get("format_version") != FORMAT_VERSION:
            raise RuntimeError("Old z-score checkpoints cannot resume with the v2 policy")
        for key in (
            "state_dim",
            "horizon",
            "n_obs_steps",
            "n_action_steps",
            "num_inference_steps",
            "batch_size",
            "lr",
            "weight_decay",
            "lr_warmup_steps",
            "val_ratio",
            "seed",
            "num_workers",
            "image_width",
            "image_height",
            "amp",
            "ema",
            "freeze_backbone",
            "pretrained_backbone",
        ):
            setattr(args, key, saved_config[key])
        args.camera_names = ",".join(saved_config["camera_names"])
        if args.dataset_dir is None:
            args.dataset_dir = saved_config["dataset_dir"]
        if args.num_epochs is None:
            args.num_epochs = saved_config["num_epochs"]
    else:
        args.dataset_dir = args.dataset_dir or str(DEFAULT_DATASET_DIR)
        args.num_epochs = args.num_epochs or 50

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        message = f"Dataset directory not found: {dataset_dir}"
        if args.resume:
            message += "; pass its current location with --dataset-dir"
        raise FileNotFoundError(message)
    args.dataset_dir = str(dataset_dir)
    seed_everything(args.seed)
    episodes = list_episodes(dataset_dir)
    if len(episodes) < 2:
        raise RuntimeError(f"At least two HDF5 episodes are required in {dataset_dir}")

    camera_names = [name.strip() for name in args.camera_names.split(",") if name.strip()]

    by_id = {episode_id(path): path for path in episodes}
    if saved_config is not None:
        expected_model = model_config(args, camera_names)
        if saved_config["model"] != expected_model:
            raise ValueError("Resume model settings differ from train_config.json")
        train_paths = [by_id[index] for index in saved_config["train_episode_ids"]]
        val_paths = [by_id[index] for index in saved_config["val_episode_ids"]]
        with stats_path.open("rb") as source:
            stats = NormalizationStats.from_dict(pickle.load(source))
    else:
        order = np.random.default_rng(args.seed).permutation(len(episodes))
        val_count = max(1, int(round(len(episodes) * args.val_ratio)))
        val_paths = [episodes[index] for index in order[:val_count]]
        train_paths = [episodes[index] for index in order[val_count:]]
        stats = calculate_stats(train_paths, args.state_dim)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        atomic_pickle_save(stats.as_dict(), stats_path)

    for index, joint in enumerate(JOINT_NAMES):
        lower, upper = JOINT_LIMITS[joint]
        if stats.action_min[index] < lower or stats.action_max[index] > upper:
            print(
                f"Warning: recorded {joint} range [{stats.action_min[index]:.5f}, "
                f"{stats.action_max[index]:.5f}] exceeds configured [{lower:.5f}, {upper:.5f}]"
            )

    dataset_kwargs = {
        "camera_names": camera_names,
        "state_dim": args.state_dim,
        "horizon": args.horizon,
        "n_obs_steps": args.n_obs_steps,
        "n_action_steps": args.n_action_steps,
        "image_size": (args.image_width, args.image_height),
    }
    train_dataset = ScaraDiffusionDataset(train_paths, **dataset_kwargs)
    val_dataset = ScaraDiffusionDataset(val_paths, **dataset_kwargs)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    first_sample = train_dataset[0]
    image_shapes = {name: list(first_sample[name].shape[1:]) for name in camera_names}
    config = vars(args) | {
        "format_version": FORMAT_VERSION,
        "normalization": "limits_-1_1",
        "action_alignment": "upstream_sequence_v1",
        "camera_names": camera_names,
        "image_shapes": image_shapes,
        "train_episode_ids": [episode_id(path) for path in train_paths],
        "val_episode_ids": [episode_id(path) for path in val_paths],
        "model": model_config(args, camera_names),
    }
    if saved_config is None:
        atomic_write_text(config_path, json.dumps(config, indent=2) + "\n")
    else:
        config = saved_config | {
            "dataset_dir": args.dataset_dir,
            "ckpt_dir": str(ckpt_dir),
            "num_epochs": args.num_epochs,
            "device": args.device,
            "resume": True,
        }
        atomic_write_text(config_path, json.dumps(config, indent=2) + "\n")

    constructor_config = dict(config["model"])
    if args.resume:
        constructor_config["pretrained_backbone"] = False
    model = ScaraDiffusionPolicy(**constructor_config)
    model.set_normalizer(stats)
    ema_model = copy.deepcopy(model) if args.ema else None
    model.to(device)
    if ema_model is not None:
        ema_model.to(device)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = min(
        len(train_loader),
        args.max_train_batches if args.max_train_batches is not None else len(train_loader),
    )
    total_steps = batches_per_epoch * args.num_epochs
    warmup_steps = min(args.lr_warmup_steps, max(0, total_steps - 1))
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    ema = (
        EMAModel(
            ema_model,
            update_after_step=0,
            inv_gamma=1.0,
            power=0.75,
            min_value=0.0,
            max_value=0.9999,
        )
        if ema_model is not None
        else None
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch, global_step, best_val_loss = 0, 0, float("inf")

    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("format_version") != FORMAT_VERSION:
            raise RuntimeError("Resume state belongs to an incompatible policy version")
        model.load_state_dict(state["model"])
        if ema_model is not None:
            if state["ema_model"] is None:
                raise ValueError("This run was trained without EMA")
            ema_model.load_state_dict(state["ema_model"])
            ema.optimization_step = int(state.get("ema_optimization_step", state["global_step"]))
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_val_loss = float(state["best_val_loss"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resumed {ckpt_dir} at epoch {start_epoch}, optimizer step {global_step}")
    else:
        # Make a new SLURM run resumable before the potentially long first epoch.
        save_training_state(
            state_path,
            epoch=-1,
            global_step=global_step,
            best_val_loss=best_val_loss,
            model=model,
            ema_model=ema_model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            config=config,
        )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Data: {len(train_paths)} train / {len(val_paths)} val episodes")
    print(f"Windows: {len(train_dataset)} train / {len(val_dataset)} val")
    print(f"Parameters: {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total")

    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model.compute_loss(move_batch(batch, device))
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = not use_amp or scaler.get_scale() >= old_scale
            if optimizer_stepped:
                lr_scheduler.step()
                if ema is not None:
                    ema.step(model)
                global_step += 1
            train_losses.append(float(loss.detach().cpu()))

        if not train_losses:
            raise RuntimeError("training loader produced no batches")
        validation_model = ema_model if ema_model is not None else model
        val_loss = mean_loss(
            validation_model,
            val_loader,
            device,
            args.max_val_batches,
            use_amp,
            seed=args.seed + 10_000,
        )
        train_loss = float(np.mean(train_losses))
        lr = lr_scheduler.get_last_lr()[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        else:
            peak_memory_gb = 0.0
        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch} | train noise MSE {train_loss:.6f} | "
            f"val noise MSE {val_loss:.6f} | lr {lr:.2e} | "
            f"time {epoch_seconds:.1f}s | peak GPU {peak_memory_gb:.2f} GB"
        )

        inference_model = ema_model if ema_model is not None else model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            atomic_torch_save(inference_model.state_dict(), ckpt_dir / "policy_best.ckpt")
        save_training_state(
            state_path,
            epoch,
            global_step,
            best_val_loss,
            model,
            ema_model,
            ema,
            optimizer,
            lr_scheduler,
            scaler,
            config,
        )
        atomic_torch_save(inference_model.state_dict(), ckpt_dir / "policy_last.ckpt")
        if (epoch + 1) % args.save_every == 0:
            atomic_torch_save(
                inference_model.state_dict(), ckpt_dir / f"policy_epoch_{epoch + 1}.ckpt"
            )


if __name__ == "__main__":
    main()
