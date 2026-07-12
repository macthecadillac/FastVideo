# SPDX-License-Identifier: Apache-2.0
"""Fake-model tests for the TDM training method."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch
from torch.testing import assert_close

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


class _ShiftedFlowScheduler(_FakeFlowScheduler):

    shift = 8.0

    def __init__(self) -> None:
        raw_sigmas = torch.tensor(
            [1.0, 0.75, 0.5, 0.25, 0.0],
            dtype=torch.float32,
        )
        shifted_sigmas = self.shift * raw_sigmas / (1.0 + (self.shift - 1.0) * raw_sigmas)
        self.timesteps = shifted_sigmas * 1000.0
        self.sigmas = shifted_sigmas
        self.config = SimpleNamespace(
            num_train_timesteps=1000,
            use_dynamic_shifting=False,
            shift_terminal=None,
            use_karras_sigmas=False,
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
        )


class _DenseShiftedFlowScheduler(_ShiftedFlowScheduler):

    def __init__(self) -> None:
        raw_timesteps = torch.arange(
            1000,
            0,
            -1,
            dtype=torch.float32,
        )
        raw_sigmas = raw_timesteps / 1000.0
        shifted_sigmas = self.shift * raw_sigmas / (1.0 + (self.shift - 1.0) * raw_sigmas)
        self.timesteps = shifted_sigmas * 1000.0
        self.sigmas = shifted_sigmas
        self.config = SimpleNamespace(
            num_train_timesteps=1000,
            use_dynamic_shifting=False,
            shift_terminal=None,
            use_karras_sigmas=False,
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
        )


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
        self.predict_calls: list[tuple[bool, str, float, bool]] = []

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
        del batch, cfg_uncond
        timestep_value = float(timestep.reshape(-1)[0].item())
        self.predict_calls.append((
            conditional,
            attn_kind,
            timestep_value,
            torch.is_grad_enabled(),
        ))
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
    generator_timestep = float(torch.as_tensor(metrics["tdm/generator/timestep"]).item())
    generator_sigma = float(torch.as_tensor(metrics["tdm/generator/sigma"]).item())
    assert generator_timestep in {750.0, 500.0}
    assert generator_sigma in {0.75, 0.5}
    assert "tdm/generator/raw_delta_abs_mean" in metrics
    assert "tdm/generator/target_delta_abs_mean" in metrics
    assert "tdm/generator/normalization_denom" in metrics
    assert metrics["tdm/generator/trajectory_index"] in {1.0, 2.0}
    assert metrics["tdm/generator/normalize_delta"] == 1.0
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

    grad_student_calls = [
        call for call in student.predict_calls
        if call[0] is True and call[1] == "vsa" and call[3] is True
    ]
    assert len(grad_student_calls) == 1
    assert grad_student_calls[0][2] in {750.0, 500.0}


def test_tdm_generator_loss_samples_separate_mode_target_step_list() -> None:
    method, _, _ = _build_method()
    valid_steps = {750, 500}

    for _ in range(16):
        timestep = method._sample_training_timestep(torch.device("cpu"))

        assert int(timestep.item()) in valid_steps


def test_tdm_generator_loss_samples_highest_non_terminal_step_in_terminal_mode() -> None:
    method, _, _ = _build_method(method_overrides={"noise_interval_mode": "to_terminal"})

    for _ in range(16):
        timestep = method._sample_training_timestep(torch.device("cpu"))

        assert int(timestep.item()) == 750


def test_tdm_sigma_lookup_treats_explicit_steps_as_raw_wan_timesteps() -> None:
    method, student, _ = _build_method()
    student.noise_scheduler = _ShiftedFlowScheduler()

    sigmas = method._timestep_to_sigma(torch.tensor([1000, 750, 500, 250]))

    expected = torch.tensor([
        1.0,
        8.0 * 0.75 / (1.0 + 7.0 * 0.75),
        8.0 * 0.5 / (1.0 + 7.0 * 0.5),
        8.0 * 0.25 / (1.0 + 7.0 * 0.25),
    ])
    assert_close(sigmas, expected)


def test_tdm_warped_denoising_steps_use_scheduler_sigmas_without_double_shift() -> None:
    method, student, _ = _build_method(method_overrides={"warp_denoising_step": True})
    student.noise_scheduler = _DenseShiftedFlowScheduler()

    steps = method._get_denoising_step_list(torch.device("cpu"))
    sigmas = method._timestep_to_sigma(steps)
    step_indices = torch.tensor([0, 250, 500, 750])
    expected_steps = student.noise_scheduler.timesteps[step_indices]
    expected_sigmas = student.noise_scheduler.sigmas[step_indices]

    assert_close(steps, expected_steps)
    assert_close(sigmas, expected_sigmas)


@pytest.mark.parametrize(
    ("steps", "match"),
    [
        ([1000], "at least two"),
        ([750, 500], "terminal sigma"),
        ([1000, 750, 750], "strictly decreasing"),
        ([1000, 500, 750], "strictly decreasing"),
    ],
)
def test_tdm_rejects_invalid_denoising_step_schedules(
    steps: list[int],
    match: str,
) -> None:
    method, _, _ = _build_method(method_overrides={"tdm_denoising_steps": steps})

    with pytest.raises(ValueError, match=match):
        method._get_denoising_step_list(torch.device("cpu"))


def test_tdm_respects_generator_update_interval() -> None:
    method, student, critic = _build_method(generator_update_interval=2)

    loss_map, outputs, metrics = method.single_train_step({}, iteration=1)

    assert metrics["update_student"] == 0.0
    assert "tdm/generator/timestep" not in metrics
    assert outputs["_fv_backward"]["update_student"] is False

    method.backward(loss_map, outputs)

    assert student.backward_calls == 0
    assert critic.backward_calls == 1


def test_tdm_separate_noise_interval_excludes_terminal_sigma() -> None:
    method, _, _ = _build_method(method_overrides={"noise_interval_mode": "separate"})
    batch = method.student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    trajectory = method._student_trajectory(batch)
    max_sigma = trajectory.sigmas.max().item()

    for _ in range(16):
        context = method._sample_tdm_context(trajectory)

        assert context.sigma_to.item() < max_sigma - 1e-6
        assert context.sigma_to.item() > context.sigma_from.item()
        assert context.trajectory_index in {2, 3}
        assert context.target_trajectory_index in {1, 2}


@pytest.mark.parametrize(
    ("sigma", "expected_weight"),
    [
        (0.96, 1.0 / 576.0),
        (8.0 / 9.0, 1.0 / 64.0),
        (8.0 / 11.0, 9.0 / 64.0),
        (0.25, 5.0),
    ],
)
def test_tdm_fake_score_uses_clipped_flow_snr_directly(
    sigma: float,
    expected_weight: float,
) -> None:
    method, _, _ = _build_method()
    context = SimpleNamespace(
        sigma_to=torch.tensor([sigma], dtype=torch.float32),
        mixed_noise=torch.zeros(2, 1, dtype=torch.float32),
        proposal_noise=torch.zeros(2, 1, dtype=torch.float32),
    )

    components = method._tdm_fake_score_weight_components(context)

    assert_close(
        components["snr_weight"],
        torch.tensor([expected_weight], dtype=torch.float32),
    )
    assert_close(
        components["weights"],
        torch.full((2, ), expected_weight, dtype=torch.float32),
    )


def test_tdm_to_terminal_noise_interval_targets_highest_non_terminal_sigma() -> None:
    method, _, _ = _build_method(method_overrides={
        "noise_interval_mode": "to_terminal",
        "use_randmid": False,
    })
    batch = method.student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    trajectory = method._student_trajectory(batch)

    context = method._sample_tdm_context(trajectory)

    assert context.target_trajectory_index == 1
    assert context.sigma_to.item() == 0.75
    assert context.sigma_to.item() < trajectory.sigmas.max().item()
    assert context.sigma_to.item() > context.sigma_from.item()
    assert bool((method._tdm_fake_score_weights(context) > 0).all().item())


def test_tdm_to_terminal_fake_score_loss_trains_critic() -> None:
    method, _, critic = _build_method(
        generator_update_interval=2,
        method_overrides={
            "noise_interval_mode": "to_terminal",
            "use_randmid": False,
        },
    )

    loss_map, outputs, metrics = method.single_train_step({}, iteration=1)

    assert float(loss_map["fake_score_loss"].item()) > 0.0
    assert float(torch.as_tensor(metrics["tdm/fake_score/weight_mean"]).item()) > 0.0

    method.backward(loss_map, outputs)

    grad = critic.transformer.weight.grad
    assert grad is not None
    assert float(grad.abs().item()) > 0.0
