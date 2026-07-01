# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TDM's Wan flow-matching scheduler bridge."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from fastvideo.train.methods.distribution_matching.tdm import (
    flow_effective_noise,
    flow_snr,
    flow_transition_to_noisier_sigma,
)


def test_flow_transition_reconstructs_noisier_point_for_video_latents() -> None:
    generator = torch.Generator(device="cpu").manual_seed(0)
    clean = torch.randn(2, 3, 1, 4, 4, generator=generator)
    eps_from = torch.randn(clean.shape, generator=generator)
    proposal = torch.randn(clean.shape, generator=generator)
    sigma_from = torch.tensor([0.25])
    sigma_to = torch.tensor([0.75])
    noisy_from = (1.0 - sigma_from) * clean + sigma_from * eps_from

    noisy_to, mixed_noise, beta = flow_transition_to_noisier_sigma(
        noisy_from=noisy_from,
        clean_latents=clean,
        eps_from=eps_from,
        sigma_from=sigma_from,
        sigma_to=sigma_to,
        proposal_noise=proposal,
    )

    reconstructed = (1.0 - sigma_to) * clean + sigma_to * mixed_noise
    assert_close(noisy_to, reconstructed, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(noisy_to).all()
    assert torch.isfinite(mixed_noise).all()
    assert torch.isfinite(beta).all()


def test_flow_transition_raises_for_lower_sigma_target() -> None:
    clean = torch.zeros(1, 1, 1, 2, 2)
    noisy_from = torch.ones_like(clean) * 0.5
    eps_from = torch.ones_like(clean)
    proposal = torch.randn_like(clean)

    with pytest.raises(ValueError, match="sigma_to >= sigma_from"):
        flow_transition_to_noisier_sigma(
            noisy_from=noisy_from,
            clean_latents=clean,
            eps_from=eps_from,
            sigma_from=torch.tensor([0.75]),
            sigma_to=torch.tensor([0.25]),
            proposal_noise=proposal,
        )


def test_flow_effective_noise_and_snr_match_wan_parameterization() -> None:
    clean = torch.tensor([[[[[2.0]]]]])
    eps = torch.tensor([[[[[-1.0]]]]])
    sigma = torch.tensor([0.2])
    noisy = (1.0 - sigma) * clean + sigma * eps

    recovered = flow_effective_noise(noisy, clean, sigma)
    assert_close(recovered, eps)
    assert_close(flow_snr(sigma), torch.tensor([16.0]))
