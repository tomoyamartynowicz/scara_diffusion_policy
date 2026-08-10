from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
from torch.utils.data import default_collate

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.scara_image_dataset import ScaraImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class ScaraImageRunner(BaseImageRunner):
    """Safe, offline evaluation on held-out SCARA demonstrations.

    This first integration deliberately never connects to the camera or robot.
    It reports held-out action errors so Stanford's training workspace can rank
    checkpoints without moving hardware during training.
    """

    def __init__(
        self,
        output_dir: str,
        shape_meta: Mapping,
        dataset_path: str,
        horizon: int,
        pad_before: int,
        pad_after: int,
        n_obs_steps: int,
        n_action_steps: int,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: Optional[int] = None,
        n_test_samples: int = 16,
        batch_size: int = 8,
    ) -> None:
        super().__init__(output_dir)
        if n_test_samples < 1 or batch_size < 1:
            raise ValueError("n_test_samples and batch_size must be positive")

        dataset = ScaraImageDataset(
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            n_obs_steps=n_obs_steps,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
        )
        self.dataset = dataset.get_validation_dataset()
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.batch_size = int(batch_size)

        count = min(int(n_test_samples), len(self.dataset))
        self.sample_indices = (
            np.linspace(0, len(self.dataset) - 1, num=count, dtype=np.int64).tolist()
            if count
            else []
        )

    def run(self, policy: BaseImagePolicy) -> dict[str, float]:
        if not self.sample_indices:
            return {
                "test_action_mse": 0.0,
                "test_normalized_action_mse": 0.0,
                "test_mean_score": 0.0,
            }

        policy.reset()
        physical_squared_error = 0.0
        normalized_squared_error = 0.0
        value_count = 0
        action_normalizer = policy.normalizer["action"]

        with torch.no_grad():
            for offset in range(0, len(self.sample_indices), self.batch_size):
                indices = self.sample_indices[offset : offset + self.batch_size]
                batch = default_collate([self.dataset[index] for index in indices])
                obs = dict_apply(
                    batch["obs"],
                    lambda value: value.to(device=policy.device, non_blocking=True),
                )
                target = batch["action"].to(device=policy.device, non_blocking=True)

                prediction = policy.predict_action(obs)["action"]
                target_start = self.n_obs_steps - 1
                target_end = target_start + self.n_action_steps
                target = target[:, target_start:target_end]
                if prediction.shape != target.shape:
                    raise RuntimeError(
                        f"Policy returned {tuple(prediction.shape)}, expected {tuple(target.shape)}"
                    )
                if not torch.isfinite(prediction).all():
                    raise RuntimeError("Policy returned NaN or Inf actions")

                physical_squared_error += torch.square(prediction - target).sum().item()
                normalized_prediction = action_normalizer.normalize(prediction)
                normalized_target = action_normalizer.normalize(target)
                normalized_squared_error += torch.square(
                    normalized_prediction - normalized_target
                ).sum().item()
                value_count += target.numel()

        action_mse = physical_squared_error / value_count
        normalized_action_mse = normalized_squared_error / value_count
        return {
            "test_action_mse": action_mse,
            "test_normalized_action_mse": normalized_action_mse,
            "test_mean_score": 1.0 / (1.0 + normalized_action_mse),
        }
