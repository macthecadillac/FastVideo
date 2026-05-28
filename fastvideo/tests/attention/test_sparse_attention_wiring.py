# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch


def test_vsa_metadata_tile_size_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")

    from fastvideo.attention.backends.video_sparse_attn import (
        BSHD_VSA_TILE_SIZE,
        DEFAULT_VSA_TILE_SIZE,
        VideoSparseAttentionMetadataBuilder,
        video_sparse_attn_bshd,
    )

    builder = VideoSparseAttentionMetadataBuilder()
    device = torch.device("cuda")
    common_kwargs = {
        "current_timestep": 0,
        "raw_latent_shape": (16, 64, 64),
        "patch_size": (1, 2, 2),
        "VSA_sparsity": 0.8,
        "device": device,
    }

    default_metadata = builder.build(**common_kwargs)
    assert default_metadata.tile_size == DEFAULT_VSA_TILE_SIZE
    assert int(default_metadata.variable_block_sizes.max().item()) <= 64

    monkeypatch.setenv("FASTVIDEO_VSA_TILE_SIZE", "auto")
    auto_metadata = builder.build(**common_kwargs)
    expected_auto_tile_size = BSHD_VSA_TILE_SIZE if video_sparse_attn_bshd is not None else DEFAULT_VSA_TILE_SIZE
    assert auto_metadata.tile_size == expected_auto_tile_size

    explicit_fast_metadata = builder.build(**common_kwargs, VSA_tile_size="4,8,8")
    assert explicit_fast_metadata.tile_size == expected_auto_tile_size

    requested_metadata = builder.build(**common_kwargs, VSA_tile_size="4,4,4")
    assert requested_metadata.tile_size == DEFAULT_VSA_TILE_SIZE


def test_denoising_stage_does_not_select_generic_bsa(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")

    from fastvideo.attention.selector import _cached_get_attn_backend
    from fastvideo.pipelines.stages.denoising import DenoisingStage

    class DummyTransformer:
        hidden_size = 128
        num_attention_heads = 2

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "BSA_ATTN")
    _cached_get_attn_backend.cache_clear()
    try:
        stage = DenoisingStage(transformer=DummyTransformer(), scheduler=None)
        assert stage.attn_backend.get_name() != "BSA_ATTN"
    finally:
        _cached_get_attn_backend.cache_clear()
