# SPDX-License-Identifier: Apache-2.0
"""Configurable diffusion samplers for RL training methods."""

from __future__ import annotations

import copy
import math
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
class SamplingTrace:
    """Stochastic trajectory slices needed by GRPO-style objectives."""

    latents: torch.Tensor
    next_latents: torch.Tensor
    timesteps: torch.Tensor
    log_probs: torch.Tensor
    step_indices: torch.Tensor


@dataclass(slots=True)
class SamplingResult:
    latents: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    trace: SamplingTrace | None = None


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


def _expand_timestep_to_batch(
    timestep: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    timestep = timestep.to(device=device)
    if timestep.ndim == 0:
        return timestep.reshape(1).expand(batch_size)
    if timestep.ndim == 1 and int(timestep.shape[0]) == 1:
        return timestep.expand(batch_size)
    if timestep.ndim == 1 and int(timestep.shape[0]) == batch_size:
        return timestep
    raise ValueError(f"timestep must be scalar, [1], or [{batch_size}], got {tuple(timestep.shape)}")


def _scheduler_step_indices(
    scheduler: Any,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    schedule = scheduler.timesteps.to(device=timesteps.device)
    indices = []
    for timestep in timesteps:
        if hasattr(scheduler, "index_for_timestep"):
            try:
                index = scheduler.index_for_timestep(timestep, schedule)
            except TypeError:
                index = scheduler.index_for_timestep(timestep)
            if torch.is_tensor(index):
                index = int(index.detach().cpu().item())
            indices.append(int(index))
            continue
        indices.append(int(torch.argmin(torch.abs(schedule - timestep)).detach().cpu().item()))
    return torch.tensor(indices, device=timesteps.device, dtype=torch.long)


def _flow_schedule_terms(
    scheduler: Any,
    timestep: torch.Tensor,
    sample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    timesteps = _expand_timestep_to_batch(timestep, int(sample.shape[0]), sample.device)
    sigmas = scheduler.sigmas.to(device=sample.device, dtype=torch.float32)
    if sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("scheduler.sigmas must be a 1D tensor with at least two entries")
    step_indices = _scheduler_step_indices(scheduler, timesteps)
    if torch.any(step_indices + 1 >= sigmas.numel()):
        raise ValueError("SDE/log-prob computation requires a following sigma for every timestep")

    view_shape = (int(sample.shape[0]), ) + (1, ) * (sample.ndim - 1)
    sigma = sigmas[step_indices].reshape(view_shape)
    sigma_next = sigmas[step_indices + 1].reshape(view_shape)
    dt = sigma_next - sigma
    sigma_max = sigmas[1]
    return step_indices, sigma, sigma_next, dt, sigma_max


def flow_ode_step(
    scheduler: Any,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    """Deterministic flow-matching Euler update independent of scheduler state."""
    _, _, _, dt, _ = _flow_schedule_terms(scheduler, timestep, sample)
    prev_sample = sample.float() + dt * model_output.float()
    return prev_sample.to(dtype=model_output.dtype)


def flow_sde_step_with_logprob(
    scheduler: Any,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    *,
    noise_level: float,
    prev_sample: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flow-SDE transition with per-sample log probability.

    This mirrors GenRL's Wan SDE log-prob step but operates on FastVideo's
    scheduler/model primitives and arbitrary latent rank.
    """
    noise_level = float(noise_level)
    if noise_level <= 0.0:
        raise ValueError("noise_level must be positive for SDE log-prob computation")

    _, sigma, _, dt, sigma_max = _flow_schedule_terms(scheduler, timestep, sample)
    model_output_f = model_output.float()
    sample_f = sample.float()
    denominator_sigma = torch.where(
        sigma == 1,
        torch.as_tensor(sigma_max, device=sigma.device, dtype=sigma.dtype),
        sigma,
    )
    std_dev_t = torch.sqrt(sigma / (1.0 - denominator_sigma)) * noise_level
    sqrt_neg_dt = torch.sqrt(-dt)
    prev_sample_mean = (sample_f * (1.0 + std_dev_t.square() / (2.0 * sigma) * dt) + model_output_f *
                        (1.0 + std_dev_t.square() * (1.0 - sigma) / (2.0 * sigma)) * dt)
    std = std_dev_t * sqrt_neg_dt
    if prev_sample is None:
        variance_noise = torch.randn(
            sample.shape,
            device=sample.device,
            dtype=model_output.dtype,
            generator=generator,
        )
        prev_sample_f = prev_sample_mean + std * variance_noise.float()
    else:
        prev_sample_f = prev_sample.float()

    log_prob = (-((prev_sample_f.detach() - prev_sample_mean).square()) / (2.0 * std.square()) - torch.log(std) -
                math.log(math.sqrt(2.0 * math.pi)))
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return (
        prev_sample_f.to(dtype=model_output.dtype),
        log_prob,
        prev_sample_mean,
        std_dev_t,
        sqrt_neg_dt,
        sigma,
        sigma_max,
    )


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

    @torch.no_grad()
    def sample_with_log_probs(
        self,
        model: ModelBase,
        batch: TrainingBatch,
        *,
        generator: torch.Generator | None,
    ) -> SamplingResult:
        """Sample a trajectory and record stochastic SDE-window transitions."""
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
        use_sde_step = sde_step_mask(self.config, len(timesteps), device=current.device)

        original_timesteps = batch.timesteps
        trace_latents: list[torch.Tensor] = []
        trace_next_latents: list[torch.Tensor] = []
        trace_timesteps: list[torch.Tensor] = []
        trace_log_probs: list[torch.Tensor] = []
        trace_step_indices: list[int] = []
        try:
            for step_idx, timestep in enumerate(timesteps):
                timestep_tensor = self._model_timestep(timestep, current)
                batch.timesteps = timestep_tensor
                pred_noise = model.predict_noise(
                    current,
                    timestep_tensor,
                    batch,
                    conditional=True,
                    attn_kind="dense",
                )
                if bool(use_sde_step[step_idx].item()):
                    next_latents, log_prob, *_ = flow_sde_step_with_logprob(
                        scheduler,
                        pred_noise,
                        timestep_tensor,
                        current,
                        noise_level=self.config.sde_noise_scale,
                        generator=generator,
                    )
                    trace_latents.append(current.detach())
                    trace_next_latents.append(next_latents.detach())
                    trace_timesteps.append(timestep_tensor.detach())
                    trace_log_probs.append(log_prob.detach())
                    trace_step_indices.append(step_idx)
                    current = next_latents
                    continue

                current = flow_ode_step(
                    scheduler,
                    pred_noise,
                    timestep_tensor,
                    current,
                )
            trace: SamplingTrace | None = None
            if trace_latents:
                trace = SamplingTrace(
                    latents=torch.stack(trace_latents, dim=1),
                    next_latents=torch.stack(trace_next_latents, dim=1),
                    timesteps=torch.stack(trace_timesteps, dim=1),
                    log_probs=torch.stack(trace_log_probs, dim=1),
                    step_indices=torch.tensor(trace_step_indices, device=current.device, dtype=torch.long),
                )
            return SamplingResult(latents=current, timesteps=timesteps, sigmas=sigmas, trace=trace)
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
