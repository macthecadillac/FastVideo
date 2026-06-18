# SPDX-License-Identifier: Apache-2.0
"""Configurable diffusion samplers for RL training methods."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import torch

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.base import ModelBase

SchedulerName = Literal["flow_match_euler", "model_default"]
TrajectoryName = Literal["ode", "sde_reflow", "mixed_ode_sde"]


@dataclass(slots=True)
class SamplingConfig:
    """YAML-backed sampling knobs shared by RL methods."""

    num_steps: int = 25
    scheduler: SchedulerName = "model_default"
    trajectory: TrajectoryName = "ode"
    flow_shift: float | None = None
    timesteps: list[float] | None = None
    sigmas: list[float] | None = None
    sde_window_size: int | None = None
    sde_window_start: int = 0
    sde_noise_scale: float = 1.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> SamplingConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"method.sampling must be a mapping, got {type(raw).__name__}")
        supported_keys = {
            "flow_shift",
            "num_steps",
            "scheduler",
            "sde_noise_scale",
            "sde_window_size",
            "sde_window_start",
            "sigmas",
            "timesteps",
            "trajectory",
        }
        unsupported_keys = sorted(set(raw) - supported_keys)
        if unsupported_keys:
            raise ValueError(f"Unsupported method.sampling key(s): {unsupported_keys}. "
                             f"Supported keys: {sorted(supported_keys)}")
        scheduler = str(raw.get("scheduler", "model_default") or "model_default").strip().lower()
        if scheduler not in {"flow_match_euler", "model_default"}:
            raise ValueError("method.sampling.scheduler must be one of "
                             "{flow_match_euler, model_default}, got "
                             f"{raw.get('scheduler')!r}")
        trajectory = str(raw.get("trajectory", "ode") or "ode").strip().lower()
        if trajectory not in {"ode", "sde_reflow", "mixed_ode_sde"}:
            raise ValueError("method.sampling.trajectory must be one of "
                             "{ode, sde_reflow, mixed_ode_sde}, got "
                             f"{raw.get('trajectory')!r}")
        timesteps = raw.get("timesteps")
        sigmas = raw.get("sigmas")
        if timesteps is not None:
            if not isinstance(timesteps, list) or not timesteps:
                raise ValueError("method.sampling.timesteps must be a non-empty list when set")
            timesteps = [float(t) for t in timesteps]
        if sigmas is not None:
            if not isinstance(sigmas, list) or not sigmas:
                raise ValueError("method.sampling.sigmas must be a non-empty list when set")
            sigmas = [float(s) for s in sigmas]
        if timesteps is not None and sigmas is not None and len(timesteps) != len(sigmas):
            raise ValueError("method.sampling.timesteps and method.sampling.sigmas must have the same length")
        num_steps = int(raw.get("num_steps", 25) or 25)
        if num_steps <= 0:
            raise ValueError("method.sampling.num_steps must be positive")
        schedule_len = len(timesteps or sigmas or []) or num_steps
        sde_window_size = raw.get("sde_window_size", None)
        if sde_window_size is not None:
            sde_window_size = int(sde_window_size)
            if sde_window_size <= 0:
                raise ValueError("method.sampling.sde_window_size must be positive when set")
        elif trajectory == "mixed_ode_sde":
            sde_window_size = schedule_len
        sde_window_start = int(raw.get("sde_window_start", 0) or 0)
        if sde_window_start < 0:
            raise ValueError("method.sampling.sde_window_start must be non-negative")
        if sde_window_size is not None and sde_window_start + sde_window_size > schedule_len:
            raise ValueError("method.sampling SDE window exceeds the sampling schedule "
                             f"({sde_window_start} + {sde_window_size} > {schedule_len})")
        sde_noise_scale_raw = raw.get("sde_noise_scale", 1.0)
        sde_noise_scale = 1.0 if sde_noise_scale_raw is None else float(sde_noise_scale_raw)
        if sde_noise_scale < 0.0:
            raise ValueError("method.sampling.sde_noise_scale must be non-negative")
        return cls(
            num_steps=num_steps,
            scheduler=scheduler,  # type: ignore[arg-type]
            trajectory=trajectory,  # type: ignore[arg-type]
            flow_shift=(None if raw.get("flow_shift", None) in (None, "inherit") else float(raw["flow_shift"])),
            timesteps=timesteps,
            sigmas=sigmas,
            sde_window_size=sde_window_size,
            sde_window_start=sde_window_start,
            sde_noise_scale=sde_noise_scale,
        )


@dataclass(slots=True)
class SamplingResult:
    latents: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor


def sde_step_mask(
    config: SamplingConfig,
    schedule_len: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return which denoising steps should use SDE/log-prob optimization.

    MixGRPO uses stochastic sampling only inside a configured window and ODE
    sampling outside it. Existing trajectories map naturally to all-ODE or
    all-SDE masks so methods can share one path.
    """
    schedule_len = int(schedule_len)
    if schedule_len <= 0:
        raise ValueError("schedule_len must be positive")
    if config.trajectory == "ode":
        return torch.zeros(schedule_len, dtype=torch.bool, device=device)
    if config.trajectory == "sde_reflow":
        return torch.ones(schedule_len, dtype=torch.bool, device=device)

    window_size = int(config.sde_window_size or schedule_len)
    start = int(config.sde_window_start)
    if start < 0 or window_size <= 0 or start + window_size > schedule_len:
        raise ValueError("Invalid MixGRPO SDE window "
                         f"({start} + {window_size} for schedule_len={schedule_len})")
    mask = torch.zeros(schedule_len, dtype=torch.bool, device=device)
    mask[start:start + window_size] = True
    return mask


class DiffusionSampler:
    """Thin model/scheduler sampler used by RL methods.

    This intentionally does not call FastVideo's full inference pipelines.
    RL training needs a reusable sampling primitive that works with
    ``ModelBase`` wrappers and scheduler math without binding a method to
    model-family pipeline classes such as ``WanDMDPipeline``.
    """

    def __init__(self, config: SamplingConfig) -> None:
        self.config = config

    @torch.no_grad()
    def sample(
        self,
        model: ModelBase,
        batch: TrainingBatch,
        *,
        generator: torch.Generator | None,
    ) -> SamplingResult:
        latents = batch.latents
        if latents is None:
            raise RuntimeError("TrainingBatch.latents is required for RL sampling")
        current = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=generator,
        )

        scheduler = self._prepare_scheduler(model, current.device)
        timesteps = scheduler.timesteps.to(device=current.device)
        sigmas = scheduler.sigmas.to(device=current.device)

        original_timesteps = batch.timesteps
        try:
            return SamplingResult(
                latents=self._sample_trajectory(
                    model,
                    batch,
                    current,
                    scheduler,
                    timesteps,
                    generator=generator,
                ),
                timesteps=timesteps,
                sigmas=sigmas,
            )
        finally:
            batch.timesteps = original_timesteps

    def _prepare_scheduler(
        self,
        model: ModelBase,
        device: torch.device,
    ) -> Any:
        if self.config.scheduler == "flow_match_euler":
            shift = self.config.flow_shift
            if shift is None:
                shift = float(getattr(model.noise_scheduler, "shift", 1.0))
            scheduler = FlowMatchEulerDiscreteScheduler(shift=float(shift))
        else:
            scheduler = copy.deepcopy(model.noise_scheduler)
        kwargs: dict[str, Any] = {"device": device}
        if self.config.timesteps is not None:
            kwargs["timesteps"] = self.config.timesteps
            kwargs["num_inference_steps"] = len(self.config.timesteps)
        if self.config.sigmas is not None:
            kwargs["sigmas"] = self.config.sigmas
            kwargs["num_inference_steps"] = len(self.config.sigmas)
        if "num_inference_steps" not in kwargs:
            kwargs["num_inference_steps"] = self.config.num_steps
        scheduler.set_timesteps(**kwargs)
        return scheduler

    def _sample_trajectory(
        self,
        model: ModelBase,
        batch: TrainingBatch,
        current: torch.Tensor,
        scheduler: Any,
        timesteps: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        pred_clean = current
        use_sde_step = sde_step_mask(self.config, len(timesteps), device=current.device)
        for step_idx, timestep in enumerate(timesteps):
            timestep_tensor = self._model_timestep(timestep, current)
            batch.timesteps = timestep_tensor
            if bool(use_sde_step[step_idx].item()):
                pred_clean = model.predict_x0(
                    current,
                    timestep_tensor,
                    batch,
                    conditional=True,
                    attn_kind="dense",
                )
                if step_idx < len(timesteps) - 1:
                    next_timestep = timesteps[step_idx + 1].reshape(1).to(device=current.device)
                    noise = torch.randn(
                        pred_clean.shape,
                        device=pred_clean.device,
                        dtype=pred_clean.dtype,
                        generator=generator,
                    )
                    current = model.add_noise(pred_clean, noise * self.config.sde_noise_scale, next_timestep)
                else:
                    current = pred_clean
                continue

            pred_noise = model.predict_noise(
                current,
                timestep_tensor,
                batch,
                conditional=True,
                attn_kind="dense",
            )
            current = scheduler.step(
                pred_noise.flatten(0, 1),
                timestep,
                current.flatten(0, 1),
                return_dict=False,
            )[0].unflatten(0, pred_noise.shape[:2])
            pred_clean = current
        return pred_clean

    @staticmethod
    def _model_timestep(
        timestep: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        return timestep.reshape(1).to(device=current.device).expand(current.shape[0]).contiguous()
