from __future__ import annotations

import sys

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from scara_diffusion_policy.constants import UPSTREAM_DIR
from scara_diffusion_policy.dataset import NormalizationStats


if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # noqa: E402


def replace_batch_norm(module: nn.Module) -> None:
    """Use GroupNorm like the upstream image policy so EMA remains valid."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.GroupNorm(child.num_features // 16, child.num_features)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_batch_norm(child)


class ScaraDiffusionPolicy(nn.Module):
    def __init__(
        self,
        camera_names: list[str],
        state_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 8,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.camera_names = list(camera_names)
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.freeze_backbone = bool(freeze_backbone)

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        if self.freeze_backbone:
            backbone.requires_grad_(False)
        else:
            replace_batch_norm(backbone)
        self.vision_encoder = nn.Sequential(*list(backbone.children())[:-1])

        feature_dim = 512 * len(self.camera_names) + self.state_dim
        self.model = ConditionalUnet1D(
            input_dim=self.state_dim,
            global_cond_dim=feature_dim * self.n_obs_steps,
            diffusion_step_embed_dim=128,
            down_dims=down_dims,
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=1e-4,
            beta_end=2e-2,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
        )

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.register_buffer("qpos_scale", torch.ones(self.state_dim))
        self.register_buffer("qpos_offset", torch.zeros(self.state_dim))
        self.register_buffer("action_scale", torch.ones(self.state_dim))
        self.register_buffer("action_offset", torch.zeros(self.state_dim))

    def train(self, mode: bool = True) -> "ScaraDiffusionPolicy":
        super().train(mode)
        if self.freeze_backbone:
            self.vision_encoder.eval()
        return self

    def set_normalizer(self, stats: NormalizationStats) -> None:
        qpos_scale, qpos_offset = stats.qpos_parameters()
        action_scale, action_offset = stats.action_parameters()
        self.qpos_scale.copy_(torch.from_numpy(qpos_scale))
        self.qpos_offset.copy_(torch.from_numpy(qpos_offset))
        self.action_scale.copy_(torch.from_numpy(action_scale))
        self.action_offset.copy_(torch.from_numpy(action_offset))

    def encode_observation(
        self, batch: dict[str, torch.Tensor], normalized_qpos: torch.Tensor
    ) -> torch.Tensor:
        batch_size = normalized_qpos.shape[0]
        features = []
        for name in self.camera_names:
            image = batch[name].reshape(-1, *batch[name].shape[2:])
            image = (image - self.image_mean) / self.image_std
            feature = self.vision_encoder(image).flatten(1)
            features.append(feature.reshape(batch_size, self.n_obs_steps, -1))
        features.append(normalized_qpos)
        return torch.cat(features, dim=-1).flatten(1)

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        actions = batch["action"] * self.action_scale + self.action_offset
        qpos = batch["qpos"] * self.qpos_scale + self.qpos_offset
        noise = torch.randn(
            actions.shape, dtype=actions.dtype, device=actions.device, generator=generator
        )
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
            generator=generator,
        ).long()
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
        prediction = self.model(
            noisy_actions, timesteps, global_cond=self.encode_observation(batch, qpos)
        )
        return (prediction - noise).square().mean()

    @torch.no_grad()
    def predict_normalized_actions(
        self,
        batch: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = batch["qpos"].device
        qpos = batch["qpos"] * self.qpos_scale + self.qpos_offset
        actions = torch.randn(
            (batch["qpos"].shape[0], self.horizon, self.state_dim),
            dtype=batch["qpos"].dtype,
            device=device,
            generator=generator,
        )
        global_cond = self.encode_observation(batch, qpos)
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        for timestep in self.scheduler.timesteps:
            prediction = self.model(actions, timestep, global_cond=global_cond)
            actions = self.scheduler.step(
                prediction, timestep, actions, generator=generator
            ).prev_sample

        start = self.n_obs_steps - 1
        return actions[:, start : start + self.n_action_steps], actions

    @torch.no_grad()
    def predict_action(
        self,
        batch: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        normalized, normalized_full = self.predict_normalized_actions(batch, generator)
        action = (normalized - self.action_offset) / self.action_scale
        action_full = (normalized_full - self.action_offset) / self.action_scale
        return {"action": action, "action_pred": action_full}
