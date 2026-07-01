# SPDX-License-Identifier: Apache-2.0
"""Fake-model tests for the TDM training method."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import torch

from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.train.methods.distribution_matching.tdm import TDMMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.training_config import (
    DataConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingLoopConfig,
)


class _FakeFlowScheduler:

    num_train_timesteps = 1000

    def __init__(self) -> None:
        self.timesteps = torch.tensor(
            [1000.0, 750.0, 500.0, 250.0, 0.0],
            dtype=torch.float32,
        )
        self.sigmas = torch.tensor(
            [1.0, 0.75, 0.5, 0.25, 0.0],
            dtype=torch.float32,
        )

    def add_noise(
        self,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        t = timestep.float().reshape(-1)
        idx = torch.argmin(
            (self.timesteps.unsqueeze(0) - t.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma = self.sigmas[idx].to(device=clean_latent.device, dtype=clean_latent.dtype)
        if sigma.numel() == 1:
            sigma = sigma.reshape((1, ) + (1, ) * (clean_latent.ndim - 1))
        else:
            sigma = sigma.reshape((clean_latent.shape[0], ) + (1, ) * (clean_latent.ndim - 1))
        return (1.0 - sigma) * clean_latent + sigma * noise


class _TinyRoleModel(ModelBase):

    def __init__(
        self,
        *,
        role: str,
        trainable: bool,
    ) -> None:
        super().__init__(trainable=trainable)
        self.role = role
        self.transformer = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.transformer.weight.fill_(0.15 if trainable else 0.05)
        self.transformer.requires_grad_(trainable)
        self.noise_scheduler = _FakeFlowScheduler()
        self.backward_calls = 0
        self.predict_calls: list[tuple[bool, str]] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        del raw_batch, generator, latents_source
        latents = torch.zeros(
            2,
            2,
            1,
            2,
            2,
        )
        return TrainingBatch(
            latents=latents,
            timesteps=torch.tensor([1000]),
            attn_metadata=None,
            attn_metadata_vsa=None,
        )

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.noise_scheduler.add_noise(clean_latents, noise, timestep)

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        del timestep, batch, cfg_uncond
        self.predict_calls.append((conditional, attn_kind))
        scale = self.transformer.weight.reshape(())
        offset = {
            ("student", True): 0.05,
            ("critic", True): -0.1,
            ("teacher", True): 0.2,
            ("teacher", False): -0.2,
        }.get((self.role, conditional), 0.0)
        return noisy_latents * scale + offset

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        del ctx
        self.backward_calls += 1
        (loss / max(1, int(grad_accum_rounds))).backward()


def _build_method(
    *,
    generator_update_interval: int = 1,
    method_overrides: dict[str, Any] | None = None,
) -> tuple[TDMMethod, _TinyRoleModel, _TinyRoleModel]:
    training = TrainingConfig(
        data=DataConfig(preprocessed_data_type="text_only", seed=0),
        optimizer=OptimizerConfig(
            learning_rate=1.0e-3,
            betas=(0.0, 0.999),
            weight_decay=0.0,
            lr_scheduler="constant",
        ),
        loop=TrainingLoopConfig(max_train_steps=4),
    )
    method_cfg = {
        "rollout_mode": "simulate",
        "generator_update_interval": generator_update_interval,
        "real_score_guidance_scale": 4.5,
        "tdm_denoising_steps": [1000, 750, 500, 250],
        "student_sample_type": "sde",
        "noise_interval_mode": "separate",
        "use_randmid": True,
        "fake_score_learning_rate": 1.0e-3,
        "fake_score_betas": [0.0, 0.999],
        "fake_score_lr_scheduler": "constant",
    }
    if method_overrides:
        method_cfg.update(method_overrides)
    cfg = SimpleNamespace(
        training=training,
        method=method_cfg,
        validation={},
    )
    student = _TinyRoleModel(role="student", trainable=True)
    teacher = _TinyRoleModel(role="teacher", trainable=False)
    critic = _TinyRoleModel(role="critic", trainable=True)
    method = TDMMethod(
        cfg=cfg,
        role_models={
            "student": student,
            "teacher": teacher,
            "critic": critic,
        },
    )
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(123)
    return method, student, critic


def test_tdm_single_train_step_reports_losses_and_routes_backward() -> None:
    method, student, critic = _build_method(generator_update_interval=1)

    loss_map, outputs, metrics = method.single_train_step({}, iteration=0)

    assert set(loss_map) == {"total_loss", "generator_loss", "fake_score_loss"}
    assert bool(loss_map["total_loss"].isfinite().item())
    assert bool(loss_map["generator_loss"].isfinite().item())
    assert bool(loss_map["fake_score_loss"].isfinite().item())
    assert metrics["update_student"] == 1.0
    assert "tdm/fake_score/sigma_from" in metrics
    assert "tdm/fake_score/sigma_to" in metrics
    assert "tdm/fake_score/snr_weight" in metrics
    assert "tdm/fake_score/importance_mean" in metrics
    assert "tdm/fake_score/per_sample_loss_mean" in metrics
    assert "tdm/fake_score/trajectory_index_from" in metrics
    assert outputs["_fv_backward"]["update_student"] is True

    method.backward(loss_map, outputs)

    assert student.backward_calls == 1
    assert critic.backward_calls == 1


def test_tdm_respects_generator_update_interval() -> None:
    method, student, critic = _build_method(generator_update_interval=2)

    loss_map, outputs, metrics = method.single_train_step({}, iteration=1)

    assert metrics["update_student"] == 0.0
    assert outputs["_fv_backward"]["update_student"] is False

    method.backward(loss_map, outputs)

    assert student.backward_calls == 0
    assert critic.backward_calls == 1


def test_tdm_separate_noise_interval_excludes_terminal_sigma() -> None:
    method, _, _ = _build_method(method_overrides={"noise_interval_mode": "separate"})
    batch = method.student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    trajectory = method._student_trajectory(batch, with_grad=False)
    max_sigma = trajectory.sigmas.max().item()

    for _ in range(16):
        context = method._sample_tdm_context(trajectory)

        assert context.sigma_to.item() < max_sigma - 1e-6
        assert context.sigma_to.item() > context.sigma_from.item()
        assert context.trajectory_index in {2, 3}
        assert context.target_trajectory_index in {1, 2}


def test_tdm_to_terminal_noise_interval_targets_max_sigma() -> None:
    method, _, _ = _build_method(method_overrides={
        "noise_interval_mode": "to_terminal",
        "use_randmid": False,
    })
    batch = method.student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    trajectory = method._student_trajectory(batch, with_grad=False)

    context = method._sample_tdm_context(trajectory)

    assert context.target_trajectory_index == int(torch.argmax(trajectory.sigmas).item())
    assert context.sigma_to.item() == trajectory.sigmas.max().item()
