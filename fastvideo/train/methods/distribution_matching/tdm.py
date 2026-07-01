# SPDX-License-Identifier: Apache-2.0
"""Trajectory Distribution Matching for flow-matching video models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import (
    get_optional_float,
    require_bool,
    require_choice,
)

if TYPE_CHECKING:
    from fastvideo.pipelines import TrainingBatch


@dataclass(slots=True)
class TDMTrajectory:
    """Few-step student trajectory used by TDM losses."""

    noisy_latents: list[torch.Tensor]
    clean_latents: list[torch.Tensor]
    timesteps: torch.Tensor
    sigmas: torch.Tensor

    @property
    def final_clean(self) -> torch.Tensor:
        return self.clean_latents[-1]


@dataclass(slots=True)
class TDMSampleContext:
    """Noised trajectory point and effective noise for fake-score training."""

    clean_latents: torch.Tensor
    noisy_from: torch.Tensor
    noisy_to: torch.Tensor
    timestep_from: torch.Tensor
    timestep_to: torch.Tensor
    sigma_from: torch.Tensor
    sigma_to: torch.Tensor
    eps_from: torch.Tensor
    mixed_noise: torch.Tensor
    proposal_noise: torch.Tensor
    transition_beta: torch.Tensor
    trajectory_index: int
    target_trajectory_index: int


def _expand_sigma_for_latents(
    sigma: torch.Tensor,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Broadcast scalar, batch, or frame-level sigma tensors to *latents*."""
    sigma = sigma.to(device=latents.device, dtype=latents.dtype)
    if sigma.ndim == 0:
        sigma = sigma.reshape(1)

    if sigma.ndim == latents.ndim:
        return sigma

    if sigma.ndim == 1:
        if sigma.numel() == 1:
            return sigma.reshape((1, ) + (1, ) * (latents.ndim - 1))
        if sigma.numel() == latents.shape[0]:
            return sigma.reshape((latents.shape[0], ) + (1, ) * (latents.ndim - 1))
        if latents.ndim >= 2 and sigma.numel() == latents.shape[0] * latents.shape[1]:
            return sigma.reshape(latents.shape[0], latents.shape[1], *([1] * (latents.ndim - 2)))
        raise ValueError("Cannot broadcast sigma with shape "
                         f"{tuple(sigma.shape)} to latents {tuple(latents.shape)}")

    if sigma.ndim == 2:
        if latents.ndim < 2 or tuple(sigma.shape) != tuple(latents.shape[:2]):
            raise ValueError("Frame-level sigma shape must match latent batch/frame "
                             f"prefix, got {tuple(sigma.shape)} for {tuple(latents.shape)}")
        return sigma.reshape(*sigma.shape, *([1] * (latents.ndim - 2)))

    while sigma.ndim < latents.ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _mean_except_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 1:
        return tensor
    return tensor.mean(dim=tuple(range(1, tensor.ndim)))


def flow_effective_noise(
    noisy_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    sigma: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Recover effective flow noise from ``x_sigma = (1-sigma)x0 + sigma eps``."""
    sigma_b = _expand_sigma_for_latents(sigma, noisy_latents)
    return (noisy_latents - (1.0 - sigma_b) * clean_latents) / sigma_b.clamp_min(eps)


def flow_snr(
    sigma: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Flow-matching analogue of diffusion SNR, ``((1-sigma)/sigma)^2``."""
    sigma = sigma.float()
    return ((1.0 - sigma).clamp_min(0.0) / sigma.clamp_min(eps)).square()


def flow_transition_to_noisier_sigma(
    *,
    noisy_from: torch.Tensor,
    clean_latents: torch.Tensor,
    eps_from: torch.Tensor,
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
    proposal_noise: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move a flow-matching point from ``sigma_from`` to ``sigma_to``.

    This is the Wan/flow analogue of TDM's diffusion between-timestep noising.
    The returned ``mixed_noise`` satisfies:

    ``noisy_to == (1 - sigma_to) * clean_latents + sigma_to * mixed_noise``.
    """
    s1 = _expand_sigma_for_latents(sigma_from, noisy_from)
    s2 = _expand_sigma_for_latents(sigma_to, noisy_from)
    if bool(torch.any(s2 + eps < s1).item()):
        raise ValueError("TDM flow transition requires sigma_to >= sigma_from")
    if bool(torch.any((1.0 - s1).abs() < eps).item()):
        raise ValueError("TDM flow transition cannot start from sigma=1")
    if bool(torch.any(s2 <= eps).item()):
        raise ValueError("TDM flow transition requires sigma_to > 0")

    a = (1.0 - s2) / (1.0 - s1).clamp_min(eps)
    beta_sq = s2.square() - (a * s1).square()
    min_beta_sq = beta_sq.min()
    if bool((min_beta_sq < -1e-6).item()):
        raise ValueError("TDM flow transition produced negative beta_sq")
    beta_sq = beta_sq.clamp_min(0.0)
    beta = beta_sq.sqrt()

    noisy_to = a * noisy_from + beta * proposal_noise
    mixed_noise = (a * s1 * eps_from + beta * proposal_noise) / s2.clamp_min(eps)
    return noisy_to, mixed_noise, beta


class TDMMethod(DMD2Method):
    """Trajectory Distribution Matching adapted to Wan-style flow matching.

    TDM keeps DMD2's three-role training layout but changes the generated
    trajectory and fake-score objectives to match the reference TDM algorithm.
    Diffusion alpha-bar math is replaced with Wan's linear flow noising family:
    ``x_sigma = (1 - sigma) * x0 + sigma * eps``.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(
            cfg=cfg,
            role_models=role_models,
        )

        if self._rollout_mode != "simulate":
            raise ValueError("TDMMethod currently requires method.rollout_mode='simulate'")

        mcfg = self.method_config
        self._denoising_step_list: torch.Tensor | None = None
        self._rollout_sample_type: Literal["sde", "ode"] = require_choice(
            mcfg,
            "student_sample_type",
            {"sde", "ode"},
            default="sde",
            where="method.student_sample_type",
        )  # type: ignore[assignment]
        self._noise_interval_mode: Literal["separate", "to_terminal"] = require_choice(
            mcfg,
            "noise_interval_mode",
            {"separate", "to_terminal"},
            default="separate",
            where="method.noise_interval_mode",
        )  # type: ignore[assignment]
        self._use_randmid = require_bool(
            mcfg,
            "use_randmid",
            default=True,
            where="method.use_randmid",
        )
        self._use_huber = require_bool(
            mcfg,
            "use_huber",
            default=False,
            where="method.use_huber",
        )

        huber_c = get_optional_float(
            mcfg,
            "huber_c",
            where="method.huber_c",
        )
        if huber_c is None:
            huber_c = 0.001
        if huber_c <= 0:
            raise ValueError("method.huber_c must be positive")
        self._huber_c = float(huber_c)

        snr_clip = get_optional_float(
            mcfg,
            "snr_clip",
            where="method.snr_clip",
        )
        if snr_clip is None:
            snr_clip = 5.0
        if snr_clip <= 0:
            raise ValueError("method.snr_clip must be positive")
        self._snr_clip = float(snr_clip)

        importance_clip = get_optional_float(
            mcfg,
            "importance_weight_clip",
            where="method.importance_weight_clip",
        )
        if importance_clip is None:
            importance_clip = 10.0
        if importance_clip <= 0:
            raise ValueError("method.importance_weight_clip must be positive")
        self._importance_weight_clip = float(importance_clip)

        self._normalize_generator_delta = require_bool(
            mcfg,
            "normalize_generator_delta",
            default=True,
            where="method.normalize_generator_delta",
        )
        self._sigma_eps = 1e-8

    # TrainingMethod override: single_train_step
    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="zeros",
        )
        if training_batch.latents is None:
            raise RuntimeError("TDM requires student.prepare_batch to populate latents")

        update_student = self._should_update_student(iteration)

        generator_loss = torch.zeros(
            (),
            device=training_batch.latents.device,
            dtype=training_batch.latents.dtype,
        )
        student_ctx = None
        if update_student:
            trajectory = self._student_trajectory(training_batch, with_grad=True)
            student_ctx = (
                training_batch.timesteps,
                training_batch.attn_metadata_vsa,
            )
            generator_loss = self._tdm_generator_loss(trajectory.final_clean, training_batch)

        (
            fake_score_loss,
            critic_ctx,
            critic_outputs,
            fake_score_metrics,
        ) = self._tdm_fake_score_loss(training_batch)

        total_loss = generator_loss + fake_score_loss
        loss_map = {
            "total_loss": total_loss,
            "generator_loss": generator_loss,
            "fake_score_loss": fake_score_loss,
        }

        outputs: dict[str, Any] = dict(critic_outputs)
        outputs["_fv_backward"] = {
            "update_student": update_student,
            "student_ctx": student_ctx,
            "critic_ctx": critic_ctx,
        }
        metrics: dict[str, LogScalar] = {"update_student": float(update_student)}
        metrics.update(fake_score_metrics)
        return loss_map, outputs, metrics

    def _get_denoising_step_list(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        if (self._denoising_step_list is not None and self._denoising_step_list.device == device):
            return self._denoising_step_list

        raw = self.method_config.get("tdm_denoising_steps", None)
        if raw is None:
            raw = self.method_config.get("dmd_denoising_steps", None)
        if not isinstance(raw, list) or not raw:
            raise ValueError("method.tdm_denoising_steps must be set for TDM")

        steps = torch.tensor(
            [int(s) for s in raw],
            dtype=torch.long,
            device=device,
        )

        warp = self.method_config.get("warp_denoising_step", None)
        if warp is None:
            warp = False
        if bool(warp):
            timesteps = torch.cat((
                self.student.noise_scheduler.timesteps.to("cpu"),
                torch.tensor([0], dtype=torch.float32),
            )).to(device)
            steps = timesteps[int(self.student.num_train_timesteps) - steps].long()

        self._denoising_step_list = steps
        return steps

    def _timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        scheduler = self.student.noise_scheduler
        sigmas = scheduler.sigmas.to(device=timestep.device, dtype=torch.float32)
        timesteps = scheduler.timesteps.to(device=timestep.device, dtype=torch.float32)
        t = timestep.to(device=timestep.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.reshape(1)
        elif t.ndim == 2:
            t = t.flatten(0, 1)
        elif t.ndim != 1:
            raise ValueError(f"Invalid timestep shape: {tuple(timestep.shape)}")
        idx = torch.argmin(
            (timesteps.unsqueeze(0) - t.unsqueeze(1)).abs(),
            dim=1,
        )
        return sigmas[idx]

    def _student_trajectory(
        self,
        batch: TrainingBatch,
        *,
        with_grad: bool,
    ) -> TDMTrajectory:
        latents = batch.latents
        if latents is None:
            raise RuntimeError("TDM requires prepared batch latents")
        device = latents.device
        dtype = latents.dtype
        step_list = self._get_denoising_step_list(device)
        if len(step_list) < 2:
            raise ValueError("TDM requires at least two denoising steps")

        current_noisy = torch.randn(
            latents.shape,
            device=device,
            dtype=dtype,
            generator=self.cuda_generator,
        )
        noisy_latents: list[torch.Tensor] = []
        clean_latents: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []

        for step_idx, timestep_scalar in enumerate(step_list):
            timestep = timestep_scalar.reshape(1)
            enable_grad = with_grad and step_idx == len(step_list) - 1
            with torch.set_grad_enabled(enable_grad):
                pred_x0 = self.student.predict_x0(
                    current_noisy,
                    timestep,
                    batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="vsa",
                )

            noisy_latents.append(current_noisy)
            clean_latents.append(pred_x0)
            sigmas.append(self._timestep_to_sigma(timestep)[0])

            if step_idx + 1 >= len(step_list):
                break

            next_timestep = step_list[step_idx + 1].reshape(1)
            if self._rollout_sample_type == "sde":
                noise = torch.randn(
                    latents.shape,
                    device=device,
                    dtype=dtype,
                    generator=self.cuda_generator,
                )
                current_noisy = self.student.add_noise(pred_x0, noise, next_timestep)
            else:
                sigma_cur = self._timestep_to_sigma(timestep)
                sigma_next = self._timestep_to_sigma(next_timestep)
                eps = flow_effective_noise(
                    current_noisy,
                    pred_x0,
                    sigma_cur,
                    eps=self._sigma_eps,
                )
                sigma_next_b = _expand_sigma_for_latents(sigma_next, current_noisy)
                current_noisy = (1.0 - sigma_next_b) * pred_x0 + sigma_next_b * eps

        timesteps = step_list.to(device=device)
        sigma_tensor = torch.stack(sigmas).to(device=device)
        batch.dmd_latent_vis_dict["generator_timestep"] = timesteps[-1].float().detach()
        return TDMTrajectory(
            noisy_latents=noisy_latents,
            clean_latents=clean_latents,
            timesteps=timesteps,
            sigmas=sigma_tensor,
        )

    def _sample_tdm_context(
        self,
        trajectory: TDMTrajectory,
    ) -> TDMSampleContext:
        device = trajectory.sigmas.device
        interval_eps = 1e-6
        max_sigma = trajectory.sigmas.max()
        if self._noise_interval_mode == "to_terminal":
            eligible = torch.nonzero(
                trajectory.sigmas < max_sigma - interval_eps,
                as_tuple=False,
            ).flatten()
        else:
            source_sigmas = trajectory.sigmas.reshape(-1, 1)
            target_sigmas = trajectory.sigmas.reshape(1, -1)
            non_terminal_targets = target_sigmas < max_sigma - interval_eps
            valid_targets = (target_sigmas > source_sigmas + interval_eps) & non_terminal_targets
            eligible = torch.nonzero(
                valid_targets.any(dim=1),
                as_tuple=False,
            ).flatten()
        if eligible.numel() == 0:
            if self._noise_interval_mode == "to_terminal":
                raise ValueError("TDM trajectory has no point that can be noised to terminal sigma")
            raise ValueError("TDM separate noise interval requires at least two non-terminal trajectory sigmas")

        if self._use_randmid:
            selected_pos = torch.randint(
                0,
                eligible.numel(),
                [1],
                device=device,
                dtype=torch.long,
                generator=self.cuda_generator,
            )
            from_idx = int(eligible[selected_pos].item())
        else:
            from_idx = int(eligible[-1].item())

        sigma_from = trajectory.sigmas[from_idx].reshape(1)
        if self._noise_interval_mode == "to_terminal":
            to_idx = int(torch.argmax(trajectory.sigmas).item())
        else:
            candidates = torch.nonzero(
                (trajectory.sigmas > sigma_from[0] + interval_eps)
                & (trajectory.sigmas < max_sigma - interval_eps),
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                raise RuntimeError("TDM separate noise source has no non-terminal larger target sigma")
            to_pos = torch.randint(
                0,
                candidates.numel(),
                [1],
                device=device,
                dtype=torch.long,
                generator=self.cuda_generator,
            )
            to_idx = int(candidates[to_pos].item())

        sigma_to = trajectory.sigmas[to_idx].reshape(1)
        timestep_from = trajectory.timesteps[from_idx].reshape(1)
        timestep_to = trajectory.timesteps[to_idx].reshape(1)
        clean_latents = trajectory.clean_latents[from_idx].detach()
        noisy_from = trajectory.noisy_latents[from_idx].detach()
        eps_from = flow_effective_noise(
            noisy_from,
            clean_latents,
            sigma_from,
            eps=self._sigma_eps,
        )
        proposal_noise = torch.randn(
            clean_latents.shape,
            device=clean_latents.device,
            dtype=clean_latents.dtype,
            generator=self.cuda_generator,
        )
        noisy_to, mixed_noise, beta = flow_transition_to_noisier_sigma(
            noisy_from=noisy_from,
            clean_latents=clean_latents,
            eps_from=eps_from,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            proposal_noise=proposal_noise,
            eps=self._sigma_eps,
        )

        return TDMSampleContext(
            clean_latents=clean_latents,
            noisy_from=noisy_from,
            noisy_to=noisy_to,
            timestep_from=timestep_from,
            timestep_to=timestep_to,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            eps_from=eps_from,
            mixed_noise=mixed_noise,
            proposal_noise=proposal_noise,
            transition_beta=beta,
            trajectory_index=from_idx,
            target_trajectory_index=to_idx,
        )

    def _tdm_fake_score_loss(
        self,
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, Any, dict[str, Any], dict[str, LogScalar]]:
        with torch.no_grad():
            trajectory = self._student_trajectory(batch, with_grad=False)
            context = self._sample_tdm_context(trajectory)

        fake_x0 = self.critic.predict_x0(
            context.noisy_to,
            context.timestep_to,
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind="dense",
        )
        elementwise = self._elementwise_loss(fake_x0, context.clean_latents)
        per_sample = _mean_except_batch(elementwise)
        weight_components = self._tdm_fake_score_weight_components(context)
        weights = weight_components["weights"].to(device=per_sample.device, dtype=per_sample.dtype)
        fake_score_loss = (per_sample * weights).mean()

        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": context.clean_latents,
            "fake_score_timestep": context.timestep_to.float().detach(),
            "fake_score_source_timestep": context.timestep_from.float().detach(),
        }
        outputs = {"fake_score_latent_vis_dict": (batch.fake_score_latent_vis_dict)}
        metrics = self._tdm_fake_score_metrics(
            context,
            per_sample,
            weight_components,
        )
        return (
            fake_score_loss,
            (batch.timesteps, batch.attn_metadata),
            outputs,
            metrics,
        )

    def _tdm_fake_score_weight_components(
        self,
        context: TDMSampleContext,
    ) -> dict[str, torch.Tensor]:
        snr = flow_snr(context.sigma_to, eps=self._sigma_eps)
        snr_weight = torch.minimum(snr, torch.full_like(snr, self._snr_clip)) / snr.clamp_min(self._sigma_eps)

        mixed_sq = _mean_except_batch(context.mixed_noise.float().square())
        proposal_sq = _mean_except_batch(context.proposal_noise.float().square())
        log_importance = 0.5 * (proposal_sq - mixed_sq)
        importance = torch.exp(log_importance.clamp(
            min=-20.0,
            max=20.0,
        ))
        importance = importance.clamp(max=self._importance_weight_clip)
        weights = snr_weight.reshape(-1).mean() * importance.detach()
        return {
            "snr": snr.detach(),
            "snr_weight": snr_weight.detach(),
            "importance": importance.detach(),
            "weights": weights.detach(),
            "mixed_noise_sq": mixed_sq.detach(),
            "proposal_noise_sq": proposal_sq.detach(),
        }

    def _tdm_fake_score_weights(
        self,
        context: TDMSampleContext,
    ) -> torch.Tensor:
        return self._tdm_fake_score_weight_components(context)["weights"]

    def _tdm_fake_score_metrics(
        self,
        context: TDMSampleContext,
        per_sample_loss: torch.Tensor,
        weight_components: dict[str, torch.Tensor],
    ) -> dict[str, LogScalar]:
        weights = weight_components["weights"].float()
        importance = weight_components["importance"].float()
        per_sample_loss = per_sample_loss.detach().float()
        snr = weight_components["snr"].float()
        snr_weight = weight_components["snr_weight"].float()
        sigma_to = context.sigma_to.detach().float()
        max_sigma = torch.ones_like(sigma_to)
        terminal = torch.isclose(
            sigma_to,
            max_sigma,
            rtol=0.0,
            atol=1e-6,
        ).float()
        return {
            "tdm/fake_score/sigma_from": context.sigma_from.detach().float().mean(),
            "tdm/fake_score/sigma_to": sigma_to.mean(),
            "tdm/fake_score/timestep_from": context.timestep_from.detach().float().mean(),
            "tdm/fake_score/timestep_to": context.timestep_to.detach().float().mean(),
            "tdm/fake_score/trajectory_index_from": float(context.trajectory_index),
            "tdm/fake_score/trajectory_index_to": float(context.target_trajectory_index),
            "tdm/fake_score/sigma_to_is_terminal": terminal.mean(),
            "tdm/fake_score/snr": snr.mean(),
            "tdm/fake_score/snr_weight": snr_weight.mean(),
            "tdm/fake_score/importance_min": importance.min(),
            "tdm/fake_score/importance_mean": importance.mean(),
            "tdm/fake_score/importance_max": importance.max(),
            "tdm/fake_score/weight_min": weights.min(),
            "tdm/fake_score/weight_mean": weights.mean(),
            "tdm/fake_score/weight_max": weights.max(),
            "tdm/fake_score/per_sample_loss_min": per_sample_loss.min(),
            "tdm/fake_score/per_sample_loss_mean": per_sample_loss.mean(),
            "tdm/fake_score/per_sample_loss_max": per_sample_loss.max(),
            "tdm/fake_score/mixed_noise_sq_mean": weight_components["mixed_noise_sq"].float().mean(),
            "tdm/fake_score/proposal_noise_sq_mean": weight_components["proposal_noise_sq"].float().mean(),
            "tdm/fake_score/transition_beta_mean": context.transition_beta.detach().float().mean(),
        }

    def _sample_training_timestep(self, device: torch.device) -> torch.Tensor:
        timestep = torch.randint(
            0,
            int(self.student.num_train_timesteps),
            [1],
            device=device,
            dtype=torch.long,
            generator=self.cuda_generator,
        )
        return self.student.shift_and_clamp_timestep(timestep)

    def _tdm_generator_loss(
        self,
        generator_pred_x0: torch.Tensor,
        batch: TrainingBatch,
    ) -> torch.Tensor:
        guidance_scale = get_optional_float(
            self.method_config,
            "real_score_guidance_scale",
            where="method.real_score_guidance_scale",
        )
        if guidance_scale is None:
            guidance_scale = 1.0
        device = generator_pred_x0.device

        with torch.no_grad():
            timestep = self._sample_training_timestep(device)
            noise = torch.randn(
                generator_pred_x0.shape,
                device=device,
                dtype=generator_pred_x0.dtype,
                generator=self.cuda_generator,
            )
            noisy_latents = self.student.add_noise(generator_pred_x0.detach(), noise, timestep)
            faker_x0 = self.critic.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            real_cond_x0 = self.teacher.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            real_uncond_x0 = self.teacher.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=False,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            real_cfg_x0 = real_uncond_x0 + (real_cond_x0 - real_uncond_x0) * float(guidance_scale)
            delta = real_cfg_x0 - faker_x0
            if self._normalize_generator_delta:
                denom = torch.abs(generator_pred_x0.detach() - real_cfg_x0).mean()
                delta = delta / denom.clamp_min(self._sigma_eps)
            target = generator_pred_x0.detach() + torch.nan_to_num(delta)

        loss = self._elementwise_loss(generator_pred_x0, target)
        batch.dmd_latent_vis_dict.update({
            "dmd_timestep": timestep.float().detach(),
            "generator_pred_video": generator_pred_x0.detach(),
        })
        return 0.5 * loss.mean()

    def _elementwise_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_huber:
            return F.huber_loss(
                pred.float(),
                target.float(),
                delta=self._huber_c,
                reduction="none",
            )
        return (pred.float() - target.float()).square()
