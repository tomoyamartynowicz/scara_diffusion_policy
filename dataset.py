from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def episode_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def list_episodes(dataset_dir: str | Path) -> list[Path]:
    return sorted(Path(dataset_dir).glob("episode_*.hdf5"), key=episode_id)


def fit_dim(values: np.ndarray, state_dim: int) -> np.ndarray:
    """Require the native four-joint SCARA layout; never add zero joints."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] != state_dim:
        raise ValueError(f"expected last dimension {state_dim}, got {values.shape}")
    return values


def image_to_chw(image: np.ndarray, image_size: tuple[int, int] | None) -> np.ndarray:
    image = np.asarray(image)
    if image_size is not None and image.shape[:2] != image_size[::-1]:
        interpolation = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_AREA
        image = cv2.resize(image, image_size, interpolation=interpolation)
    if image.ndim == 2:
        image = np.repeat((image.astype(np.float32) / 65535.0)[..., None], 3, axis=-1)
    else:
        image = image.astype(np.float32) / 255.0
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)


@dataclass
class NormalizationStats:
    """Per-joint limits used by the upstream [-1, 1] linear normalizer."""

    qpos_min: np.ndarray
    qpos_max: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray

    @classmethod
    def from_dict(cls, values: dict) -> "NormalizationStats":
        required = {"qpos_min", "qpos_max", "action_min", "action_max"}
        if set(values) != required:
            raise ValueError(
                "This checkpoint uses the old z-score normalizer. "
                "Train the v2 policy in a new checkpoint directory."
            )
        return cls(**{key: np.asarray(values[key], dtype=np.float32) for key in required})

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "qpos_min": self.qpos_min,
            "qpos_max": self.qpos_max,
            "action_min": self.action_min,
            "action_max": self.action_max,
        }

    @staticmethod
    def _scale_offset(
        minimum: np.ndarray, maximum: np.ndarray, range_eps: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray]:
        value_range = maximum - minimum
        constant = value_range < range_eps
        safe_range = value_range.copy()
        safe_range[constant] = 2.0
        scale = 2.0 / safe_range
        offset = -1.0 - scale * minimum
        offset[constant] = -minimum[constant]
        return scale.astype(np.float32), offset.astype(np.float32)

    def qpos_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return self._scale_offset(self.qpos_min, self.qpos_max)

    def action_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return self._scale_offset(self.action_min, self.action_max)

    def normalize_qpos(self, values: np.ndarray) -> np.ndarray:
        scale, offset = self.qpos_parameters()
        return np.asarray(values, dtype=np.float32) * scale + offset

    def normalize_action(self, values: np.ndarray) -> np.ndarray:
        scale, offset = self.action_parameters()
        return np.asarray(values, dtype=np.float32) * scale + offset

    def unnormalize_action(self, values: np.ndarray) -> np.ndarray:
        scale, offset = self.action_parameters()
        return (np.asarray(values, dtype=np.float32) - offset) / scale


def calculate_stats(paths: list[Path], state_dim: int) -> NormalizationStats:
    if not paths:
        raise ValueError("cannot calculate normalization statistics without episodes")
    qpos_parts, action_parts = [], []
    for path in paths:
        with h5py.File(path, "r") as root:
            qpos_parts.append(fit_dim(root["observations/qpos"][()], state_dim))
            action_parts.append(fit_dim(root["action"][()], state_dim))

    qpos = np.concatenate(qpos_parts)
    action = np.concatenate(action_parts)
    return NormalizationStats(
        qpos_min=qpos.min(0).astype(np.float32),
        qpos_max=qpos.max(0).astype(np.float32),
        action_min=action.min(0).astype(np.float32),
        action_max=action.max(0).astype(np.float32),
    )


class ScaraDiffusionDataset(Dataset):
    """ACT HDF5 episodes with the original Diffusion Policy sequence padding."""

    def __init__(
        self,
        episode_paths: list[Path],
        camera_names: list[str],
        state_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.paths = list(episode_paths)
        self.camera_names = list(camera_names)
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.image_size = image_size
        self.samples: list[tuple[int, int]] = []

        if self.n_obs_steps > self.horizon:
            raise ValueError("n_obs_steps cannot exceed horizon")
        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError("n_action_steps does not fit in the prediction horizon")

        pad_before = self.n_obs_steps - 1
        pad_after = self.n_action_steps - 1
        for episode_index, path in enumerate(self.paths):
            with h5py.File(path, "r") as root:
                length = int(root["observations/qpos"].shape[0])
                if root["action"].shape != (length, self.state_dim):
                    raise ValueError(f"{path}: expected action shape {(length, self.state_dim)}")
                if root["observations/qpos"].shape != (length, self.state_dim):
                    raise ValueError(f"{path}: expected qpos shape {(length, self.state_dim)}")
                for camera_name in self.camera_names:
                    image = self._image_dataset(root, camera_name)
                    if image.shape[0] != length:
                        raise ValueError(f"{path}: camera and qpos lengths differ")

            min_start = -pad_before
            max_start = length - self.horizon + pad_after
            self.samples.extend(
                (episode_index, start) for start in range(min_start, max_start + 1)
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _image_dataset(root: h5py.File, camera_name: str):
        rgb_path = f"observations/images/{camera_name}"
        depth_path = f"observations/depth_images/{camera_name}"
        if rgb_path in root:
            return root[rgb_path]
        if depth_path in root:
            return root[depth_path]
        raise KeyError(f"camera {camera_name!r} not found")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, sequence_start = self.samples[index]
        with h5py.File(self.paths[episode_index], "r") as root:
            length = int(root["observations/qpos"].shape[0])
            obs_indices = np.clip(
                np.arange(sequence_start, sequence_start + self.n_obs_steps), 0, length - 1
            )
            action_indices = np.clip(
                np.arange(sequence_start, sequence_start + self.horizon), 0, length - 1
            )
            qpos = fit_dim(
                np.stack([root["observations/qpos"][i] for i in obs_indices]), self.state_dim
            )
            actions = fit_dim(
                np.stack([root["action"][i] for i in action_indices]), self.state_dim
            )
            images = {
                name: np.stack(
                    [
                        image_to_chw(self._image_dataset(root, name)[i], self.image_size)
                        for i in obs_indices
                    ]
                )
                for name in self.camera_names
            }

        result = {name: torch.from_numpy(value) for name, value in images.items()}
        result["qpos"] = torch.from_numpy(qpos)
        result["action"] = torch.from_numpy(actions)
        return result
