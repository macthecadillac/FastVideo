# SPDX-License-Identifier: Apache-2.0
"""Fake-model tests for the TDM training method."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
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
        self.predict_inputs: list[torch.Tensor] = []

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
        self.predict_inputs.append(noisy_latents.detach().clone())
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
    target_timestep = float(torch.as_tensor(metrics["tdm/generator/target_timestep"]).item())
    target_sigma = float(torch.as_tensor(metrics["tdm/generator/target_sigma"]).item())
    assert generator_timestep in {500.0, 250.0}
    assert generator_sigma in {0.5, 0.25}
    assert target_timestep in {750.0, 500.0}
    assert target_sigma in {0.75, 0.5}
    assert target_timestep > generator_timestep
    assert target_sigma > generator_sigma
    assert "tdm/generator/raw_delta_abs_mean" in metrics
    assert "tdm/generator/target_delta_abs_mean" in metrics
    assert "tdm/generator/normalization_denom" in metrics
    assert metrics["tdm/generator/trajectory_index"] in {2.0, 3.0}
    assert metrics["tdm/generator/target_trajectory_index"] in {1.0, 2.0}
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
    assert grad_student_calls[0][2] in {500.0, 250.0}


def test_tdm_generator_scores_target_noised_latent_at_distinct_timestep() -> None:
    method, student, critic = _build_method(method_overrides={"use_randmid": False})
    teacher = method.teacher
    batch = student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    trajectory = method._student_trajectory(batch)
    student.predict_calls.clear()
    student.predict_inputs.clear()
    critic.predict_calls.clear()
    critic.predict_inputs.clear()
    teacher.predict_calls.clear()
    teacher.predict_inputs.clear()

    _, metrics = method._tdm_generator_loss(trajectory, batch)

    source_timestep = float(torch.as_tensor(metrics["tdm/generator/timestep"]).item())
    target_timestep = float(torch.as_tensor(metrics["tdm/generator/target_timestep"]).item())
    source_index = int(float(metrics["tdm/generator/trajectory_index"]))
    assert source_timestep != target_timestep
    assert target_timestep > source_timestep
    assert student.predict_calls == [(True, "vsa", source_timestep, True)]
    assert [call[:3] for call in critic.predict_calls] == [(True, "dense", target_timestep)]
    assert [call[:3] for call in teacher.predict_calls] == [
        (True, "dense", target_timestep),
        (False, "dense", target_timestep),
    ]
    assert_close(critic.predict_inputs[0], teacher.predict_inputs[0])
    assert_close(critic.predict_inputs[0], teacher.predict_inputs[1])
    assert not torch.allclose(critic.predict_inputs[0], trajectory.noisy_latents[source_index])


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


def test_tdm_real_shifted_scheduler_aligns_noising_model_labels_and_x0_conversion() -> None:
    method, student, critic = _build_method(method_overrides={"student_sample_type": "ode"})
    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    student.noise_scheduler = scheduler
    critic.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    method.teacher.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    batch = student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")

    trajectory = method._student_trajectory(batch)

    expected_sigmas = method._timestep_to_sigma(torch.tensor([1000, 750, 500, 250]))
    model_labels = torch.tensor([call[2] for call in student.predict_calls])
    resolved_sigmas = scheduler.sigmas[
        torch.argmin((scheduler.timesteps.unsqueeze(0) - model_labels.unsqueeze(1)).abs(), dim=1)
    ]
    assert_close(trajectory.sigmas, expected_sigmas)
    assert_close(resolved_sigmas, expected_sigmas)
    assert model_labels[1].item() > 900.0

    for noisy, clean, sigma in zip(trajectory.noisy_latents, trajectory.clean_latents, trajectory.sigmas):
        predicted_flow = noisy * student.transformer.weight.reshape(()) + 0.05
        assert_close(clean, noisy - sigma * predicted_flow)

    for index in range(len(trajectory.noisy_latents) - 1):
        noisy = trajectory.noisy_latents[index]
        clean = trajectory.clean_latents[index]
        sigma = trajectory.sigmas[index]
        next_sigma = trajectory.sigmas[index + 1]
        eps = (noisy - (1.0 - sigma) * clean) / sigma
        expected_next = (1.0 - next_sigma) * clean + next_sigma * eps
        assert_close(trajectory.noisy_latents[index + 1], expected_next)


def test_tdm_warped_student_trajectory_uses_scheduler_sigmas_without_double_shift() -> None:
    method, student, _ = _build_method(method_overrides={
        "warp_denoising_step": True,
        "student_sample_type": "ode",
    })
    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    student.noise_scheduler = scheduler
    batch = student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")

    trajectory = method._student_trajectory(batch)

    step_indices = torch.tensor([0, 250, 500, 750])
    expected_steps = scheduler.timesteps[step_indices]
    expected_sigmas = scheduler.sigmas[step_indices]
    model_labels = torch.tensor([call[2] for call in student.predict_calls])
    assert_close(trajectory.timesteps, expected_steps)
    assert_close(trajectory.sigmas, expected_sigmas)
    assert_close(model_labels, expected_steps)

    for index in range(len(trajectory.noisy_latents) - 1):
        noisy = trajectory.noisy_latents[index]
        clean = trajectory.clean_latents[index]
        sigma = trajectory.sigmas[index]
        next_sigma = trajectory.sigmas[index + 1]
        eps = (noisy - (1.0 - sigma) * clean) / sigma
        expected_next = (1.0 - next_sigma) * clean + next_sigma * eps
        assert_close(trajectory.noisy_latents[index + 1], expected_next)


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


def test_tdm_generator_delta_normalization_is_per_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    method, student, critic = _build_method(method_overrides={
        "real_score_guidance_scale": 1.0,
        "use_huber": False,
    })
    teacher = method.teacher
    batch = student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    pred = torch.zeros((2, 1, 1, 1, 1), requires_grad=True)
    real_x0 = torch.tensor([1.0, 100.0]).reshape(2, 1, 1, 1, 1)
    fake_x0 = torch.zeros_like(real_x0)
    context = SimpleNamespace(
        noisy_from=torch.zeros_like(pred),
        timestep_from=torch.tensor([500.0]),
        timestep_to=torch.tensor([750.0]),
        sigma_from=torch.tensor([0.5]),
        sigma_to=torch.tensor([0.75]),
        proposal_noise=torch.zeros_like(pred),
        trajectory_index=2,
        target_trajectory_index=1,
    )

    monkeypatch.setattr(method, "_sample_tdm_context", lambda trajectory: context)
    monkeypatch.setattr(student, "predict_x0", lambda *args, **kwargs: pred)
    monkeypatch.setattr(critic, "predict_x0", lambda *args, **kwargs: fake_x0)
    monkeypatch.setattr(
        teacher,
        "predict_x0",
        lambda *args, conditional, **kwargs: real_x0 if conditional else torch.zeros_like(real_x0),
    )

    loss, metrics = method._tdm_generator_loss(SimpleNamespace(), batch)
    loss.backward()

    assert_close(pred.grad, torch.full_like(pred, -1.0))
    assert_close(
        torch.as_tensor(metrics["tdm/generator/normalization_denom"]),
        torch.tensor(50.5),
    )


def test_tdm_pseudo_huber_applies_per_sample_normalization_after_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    huber_c = 0.25
    method, student, critic = _build_method(method_overrides={
        "real_score_guidance_scale": 1.0,
        "use_huber": True,
        "huber_c": huber_c,
    })
    teacher = method.teacher
    batch = student.prepare_batch({}, generator=method.cuda_generator, latents_source="zeros")
    pred = torch.zeros((2, 1, 1, 1, 1), requires_grad=True)
    real_x0 = torch.tensor([1.0, 100.0]).reshape_as(pred)
    fake_x0 = torch.zeros_like(real_x0)
    context = SimpleNamespace(
        noisy_from=torch.zeros_like(pred),
        timestep_from=torch.tensor([500.0]),
        timestep_to=torch.tensor([750.0]),
        sigma_from=torch.tensor([0.5]),
        sigma_to=torch.tensor([0.75]),
        proposal_noise=torch.zeros_like(pred),
        trajectory_index=2,
        target_trajectory_index=1,
    )

    monkeypatch.setattr(method, "_sample_tdm_context", lambda trajectory: context)
    monkeypatch.setattr(student, "predict_x0", lambda *args, **kwargs: pred)
    monkeypatch.setattr(critic, "predict_x0", lambda *args, **kwargs: fake_x0)
    monkeypatch.setattr(
        teacher,
        "predict_x0",
        lambda *args, conditional, **kwargs: real_x0 if conditional else torch.zeros_like(real_x0),
    )

    loss, _ = method._tdm_generator_loss(SimpleNamespace(), batch)
    loss.backward()

    reference_pred = torch.zeros_like(pred, requires_grad=True)
    reference_error = reference_pred - real_x0
    reference_denom = torch.abs(reference_pred.detach() - real_x0)
    reference_loss = ((torch.sqrt(reference_error.square() + huber_c**2) - huber_c) / reference_denom).mean()
    reference_loss.backward()

    assert_close(loss, reference_loss.detach())
    assert_close(pred.grad, reference_pred.grad)
    assert not torch.isclose(pred.grad[0], pred.grad[1]).item()


def test_tdm_use_huber_is_generator_only_and_matches_reference_pseudo_huber() -> None:
    mse_method, _, _ = _build_method(method_overrides={"use_huber": False})
    huber_method, _, _ = _build_method(method_overrides={"use_huber": True, "huber_c": 0.25})
    pred = torch.tensor([0.0, 2.0])
    target = torch.tensor([1.0, 0.0])

    expected = torch.sqrt((pred - target).square() + 0.25**2) - 0.25
    assert_close(huber_method._generator_elementwise_loss(pred, target), expected)
    assert_close(mse_method._generator_elementwise_loss(pred, target), (pred - target).square())

    mse_method.cuda_generator.manual_seed(321)
    huber_method.cuda_generator.manual_seed(321)
    mse_batch = mse_method.student.prepare_batch({}, generator=mse_method.cuda_generator, latents_source="zeros")
    huber_batch = huber_method.student.prepare_batch({}, generator=huber_method.cuda_generator, latents_source="zeros")
    mse_method.cuda_generator.manual_seed(654)
    huber_method.cuda_generator.manual_seed(654)
    mse_loss, _, _, _ = mse_method._tdm_fake_score_loss(mse_batch)
    huber_loss, _, _, _ = huber_method._tdm_fake_score_loss(huber_batch)
    assert_close(huber_loss, mse_loss)
