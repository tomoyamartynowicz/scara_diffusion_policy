from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

import cv2
import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


_EPISODE_PATTERN = re.compile(r"episode_(\d+)\.hdf5$")


def _episode_number(path: Path) -> int:
    match = _EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid episode filename: {path.name}")
    return int(match.group(1))


def _get_val_mask(n_episodes: int, val_ratio: float, seed: int) -> np.ndarray:
    """Match diffusion_policy.common.sampler.get_val_mask without importing numba."""
    mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return mask
    n_validation = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    validation_indices = np.random.default_rng(seed).choice(
        n_episodes,
        size=n_validation,
        replace=False,
    )
    mask[validation_indices] = True
    return mask


def _downsample_mask(
    mask: np.ndarray,
    max_n: Optional[int],
    seed: int,
) -> np.ndarray:
    """Match diffusion_policy.common.sampler.downsample_mask."""
    if max_n is None or int(mask.sum()) <= int(max_n):
        return mask
    candidates = np.flatnonzero(mask)
    selected = np.random.default_rng(seed).choice(
        candidates,
        size=int(max_n),
        replace=False,
    )
    result = np.zeros_like(mask)
    result[selected] = True
    return result


class ScaraImageDataset(BaseImageDataset):
    """Lazy reader for the ACT-style HDF5 episodes recorded on the SCARA.

    The returned dictionary follows :class:`BaseImageDataset` exactly:

    ``obs[wrist_d405]``
        ``(T_obs, C, H, W)`` float32 RGB in ``[0, 1]``.
    ``obs[qpos]``
        ``(T_obs, 4)`` float32 joint positions.
    ``action``
        ``(horizon, 4)`` float32 absolute joint targets.

    Images stay in their HDF5 files and are read only when sampled. This is
    important for the SCARA dataset, which is much larger than system memory.
    """

    def __init__(
        self,
        shape_meta: Mapping,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.n_obs_steps = self.horizon if n_obs_steps is None else int(n_obs_steps)
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 1 <= self.n_obs_steps <= self.horizon:
            raise ValueError("n_obs_steps must be between 1 and horizon")
        if not 0 <= self.pad_before < self.horizon:
            raise ValueError("pad_before must be in [0, horizon)")
        if not 0 <= self.pad_after < self.horizon:
            raise ValueError("pad_after must be in [0, horizon)")
        if not 0.0 <= float(val_ratio) < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")

        obs_meta = shape_meta["obs"]
        self.rgb_keys = []
        self.lowdim_keys = []
        self.obs_shapes: dict[str, tuple[int, ...]] = {}
        for key, attributes in obs_meta.items():
            key = str(key)
            shape = tuple(int(value) for value in attributes["shape"])
            obs_type = str(attributes.get("type", "low_dim"))
            if obs_type == "rgb":
                if len(shape) != 3 or shape[0] not in (1, 3, 4):
                    raise ValueError(f"RGB observation {key!r} needs shape (C, H, W)")
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                if not shape:
                    raise ValueError(f"Low-dimensional observation {key!r} has no shape")
                self.lowdim_keys.append(key)
            else:
                raise ValueError(f"Unsupported observation type {obs_type!r} for {key!r}")
            self.obs_shapes[key] = shape

        self.action_shape = tuple(int(value) for value in shape_meta["action"]["shape"])
        if len(self.action_shape) != 1:
            raise ValueError("SCARA actions must be one-dimensional vectors")
        if not self.rgb_keys:
            raise ValueError("At least one RGB observation is required")
        if "qpos" not in self.lowdim_keys:
            raise ValueError("shape_meta must contain the low-dimensional qpos observation")

        episode_paths = [
            path
            for path in self.dataset_path.glob("episode_*.hdf5")
            if _EPISODE_PATTERN.fullmatch(path.name)
        ]
        self.episode_paths = sorted(episode_paths, key=_episode_number)
        if not self.episode_paths:
            raise FileNotFoundError(f"No episode_*.hdf5 files found in {self.dataset_path}")

        self.episode_lengths = np.empty(len(self.episode_paths), dtype=np.int64)
        self._validate_episodes()

        self.val_mask = _get_val_mask(
            n_episodes=len(self.episode_paths),
            val_ratio=float(val_ratio),
            seed=int(seed),
        )
        self.train_mask = _downsample_mask(
            mask=~self.val_mask,
            max_n=max_train_episodes,
            seed=int(seed),
        )
        if not np.any(self.train_mask):
            raise ValueError("The train split contains no episodes")

        self._normalizer_mask = self.train_mask.copy()
        self._normalizer_data: Optional[dict[str, np.ndarray]] = None
        self.active_mask = self.train_mask.copy()
        self.samples = self._make_samples(self.active_mask)

    def _validate_episodes(self) -> None:
        for episode_index, path in enumerate(self.episode_paths):
            with h5py.File(path, "r") as episode:
                if "action" not in episode:
                    raise KeyError(f"{path}: missing action")
                action = episode["action"]
                if action.ndim != 1 + len(self.action_shape):
                    raise ValueError(f"{path}: invalid action shape {action.shape}")
                length = int(action.shape[0])
                if length < 1 or tuple(action.shape[1:]) != self.action_shape:
                    raise ValueError(
                        f"{path}: expected action shape (T, {self.action_shape}), "
                        f"got {action.shape}"
                    )

                for key in self.lowdim_keys:
                    hdf5_key = f"observations/{key}"
                    if hdf5_key not in episode:
                        raise KeyError(f"{path}: missing {hdf5_key}")
                    expected = (length,) + self.obs_shapes[key]
                    if episode[hdf5_key].shape != expected:
                        raise ValueError(
                            f"{path}: expected {hdf5_key} shape {expected}, "
                            f"got {episode[hdf5_key].shape}"
                        )

                for key in self.rgb_keys:
                    hdf5_key = f"observations/images/{key}"
                    if hdf5_key not in episode:
                        raise KeyError(f"{path}: missing {hdf5_key}")
                    image_shape = episode[hdf5_key].shape
                    expected_channels = self.obs_shapes[key][0]
                    if (
                        len(image_shape) != 4
                        or image_shape[0] != length
                        or image_shape[-1] != expected_channels
                    ):
                        raise ValueError(
                            f"{path}: expected {hdf5_key} as (T, H, W, {expected_channels}), "
                            f"got {image_shape}"
                        )
                self.episode_lengths[episode_index] = length

    def _make_samples(self, episode_mask: np.ndarray) -> list[tuple[int, int]]:
        samples: list[tuple[int, int]] = []
        for episode_index in np.flatnonzero(episode_mask):
            length = int(self.episode_lengths[episode_index])
            min_start = -self.pad_before
            max_start = length - self.horizon + self.pad_after
            samples.extend(
                (int(episode_index), start)
                for start in range(min_start, max_start + 1)
            )
        return samples

    @staticmethod
    def _read_padded(dataset: h5py.Dataset, start: int, length: int) -> np.ndarray:
        source_start = max(start, 0)
        source_end = min(start + length, int(dataset.shape[0]))
        if source_start >= source_end:
            raise IndexError("Requested sequence does not overlap the episode")

        data = np.asarray(dataset[source_start:source_end])
        left = source_start - start
        right = start + length - source_end
        parts = []
        if left:
            parts.append(np.repeat(data[:1], left, axis=0))
        parts.append(data)
        if right:
            parts.append(np.repeat(data[-1:], right, axis=0))
        return np.concatenate(parts, axis=0) if len(parts) > 1 else data

    def _convert_images(self, images: np.ndarray, key: str) -> np.ndarray:
        channels, target_height, target_width = self.obs_shapes[key]
        converted = []
        for image in images:
            if image.shape[:2] != (target_height, target_width):
                shrinking = image.shape[0] > target_height or image.shape[1] > target_width
                interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
                image = cv2.resize(
                    image,
                    (target_width, target_height),
                    interpolation=interpolation,
                )
            if image.ndim == 2:
                image = image[..., None]
            if image.shape[-1] != channels:
                raise ValueError(f"Unexpected channel count for {key!r}: {image.shape}")
            converted.append(np.moveaxis(image, -1, 0))
        return np.ascontiguousarray(np.stack(converted), dtype=np.float32) / 255.0

    def get_validation_dataset(self) -> "ScaraImageDataset":
        validation = copy.copy(self)
        validation.active_mask = self.val_mask.copy()
        validation.samples = validation._make_samples(validation.active_mask)
        return validation

    def _get_normalizer_data(self) -> dict[str, np.ndarray]:
        if self._normalizer_data is None:
            parts: dict[str, list[np.ndarray]] = {
                "action": [],
                **{key: [] for key in self.lowdim_keys},
            }
            for episode_index in np.flatnonzero(self._normalizer_mask):
                with h5py.File(self.episode_paths[int(episode_index)], "r") as episode:
                    parts["action"].append(np.asarray(episode["action"], dtype=np.float32))
                    for key in self.lowdim_keys:
                        parts[key].append(
                            np.asarray(episode[f"observations/{key}"], dtype=np.float32)
                        )
            self._normalizer_data = {
                key: np.concatenate(values, axis=0) for key, values in parts.items()
            }
        return self._normalizer_data

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer.fit(
            data=self._get_normalizer_data(),
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._get_normalizer_data()["action"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        episode_index, start = self.samples[index]
        with h5py.File(self.episode_paths[episode_index], "r") as episode:
            obs: dict[str, torch.Tensor] = {}
            for key in self.rgb_keys:
                images = self._read_padded(
                    episode[f"observations/images/{key}"],
                    start=start,
                    length=self.n_obs_steps,
                )
                obs[key] = torch.from_numpy(self._convert_images(images, key))
            for key in self.lowdim_keys:
                values = self._read_padded(
                    episode[f"observations/{key}"],
                    start=start,
                    length=self.n_obs_steps,
                )
                obs[key] = torch.from_numpy(
                    np.ascontiguousarray(values, dtype=np.float32)
                )
            action = self._read_padded(
                episode["action"],
                start=start,
                length=self.horizon,
            )

        return {
            "obs": obs,
            "action": torch.from_numpy(np.ascontiguousarray(action, dtype=np.float32)),
        }
