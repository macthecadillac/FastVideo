# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the shipped Wan TDM example config."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.testing import assert_close

from fastvideo.train.methods.distribution_matching.tdm import (
    TDMMethod,
    flow_transition_to_noisier_sigma,
)
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.instantiate import resolve_target


def test_tdm_wan_lora_config_resolves_without_loading_weights() -> None:
    config_path = Path("examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml")

    cfg = load_run_config(str(config_path))

    assert resolve_target(str(cfg.method["_target_"])) is TDMMethod
    assert set(cfg.models) == {"student", "teacher", "critic"}
    assert cfg.models["student"]["trainable"] is True
    assert cfg.models["teacher"]["trainable"] is False
    assert cfg.models["critic"]["trainable"] is True
    assert cfg.models["student"]["lora"]["enable"] is True
    assert cfg.models["critic"]["lora"]["enable"] is True

    assert cfg.method["rollout_mode"] == "simulate"
    assert cfg.method["tdm_denoising_steps"] == [1000, 750, 500, 250]
    assert cfg.method["noise_interval_mode"] == "separate"
    assert cfg.method["student_sample_type"] == "sde"
    assert "enable_gradient_in_rollout" not in cfg.method
    assert cfg.training.pipeline_config is not None
    assert getattr(cfg.training.pipeline_config, "flow_shift") == 8


def test_tdm_flow_bridge_smoke() -> None:
    clean = torch.ones(1, 1, 1, 1, 1)
    eps_from = torch.zeros_like(clean)
    proposal = torch.ones_like(clean)
    sigma_from = torch.tensor([0.25])
    sigma_to = torch.tensor([0.5])
    noisy_from = (1.0 - sigma_from) * clean + sigma_from * eps_from

    noisy_to, mixed_noise, _ = flow_transition_to_noisier_sigma(
        noisy_from=noisy_from,
        clean_latents=clean,
        eps_from=eps_from,
        sigma_from=sigma_from,
        sigma_to=sigma_to,
        proposal_noise=proposal,
    )

    assert_close(noisy_to, (1.0 - sigma_to) * clean + sigma_to * mixed_noise)
