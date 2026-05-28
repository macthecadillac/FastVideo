# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import socket
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SP_WORLD_SIZE = 1
SEED = 2027


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _qkv(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    batch, seq_len, num_heads, head_dim = 1, 19, 4, 32
    q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, generator=generator)
    k = torch.randn(batch, seq_len, num_heads, head_dim, device=device, generator=generator)
    v = torch.randn(batch, seq_len, num_heads, head_dim, device=device, generator=generator)
    return q, k, v


def _rope(device: torch.device, seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(head_dim // 2, device=device, dtype=torch.float32).unsqueeze(0)
    angles = positions / (10000 ** (2 * freqs / head_dim))
    return torch.cos(angles), torch.sin(angles)


def _build_attention_modules():
    from fastvideo.attention import DistributedAttention, LocalAttention
    from fastvideo.attention.selector import _cached_get_attn_backend
    from fastvideo.platforms import AttentionBackendEnum

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    _cached_get_attn_backend.cache_clear()
    kwargs = {
        "num_heads": 4,
        "head_size": 32,
        "causal": False,
        "supported_attention_backends": (AttentionBackendEnum.TORCH_SDPA, ),
    }
    return DistributedAttention(prefix="tests.compile_sp1", **kwargs), LocalAttention(**kwargs)


def _assert_single_rank_dense_path_avoids_collectives(device: torch.device) -> None:
    from fastvideo.attention import layer as attention_layer
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    distributed_attention, local_attention = _build_attention_modules()
    q, k, v = _qkv(device)
    original_seq_len = q.shape[1] - 3
    freqs_cis = _rope(device, original_seq_len, q.shape[-1])

    def fail_all_to_all(*args, **kwargs):
        raise AssertionError("single-rank dense attention path unexpectedly called SP all-to-all")

    original_all_to_all = attention_layer.sequence_model_parallel_all_to_all_4D
    attention_layer.sequence_model_parallel_all_to_all_4D = fail_all_to_all
    try:
        forward_batch = ForwardBatch(data_type="dummy")
        with torch.inference_mode(), set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch,
        ):
            output, replicated_output = distributed_attention(
                q,
                k,
                v,
                original_seq_len=original_seq_len,
                freqs_cis=freqs_cis,
            )
            expected = local_attention(
                q[:, :original_seq_len],
                k[:, :original_seq_len],
                v[:, :original_seq_len],
                freqs_cis=freqs_cis,
            )
    finally:
        attention_layer.sequence_model_parallel_all_to_all_4D = original_all_to_all

    assert replicated_output is None
    expected = F.pad(expected, (0, 0, 0, 0, 0, q.shape[1] - original_seq_len))
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def _assert_single_rank_dense_path_compiles_fullgraph(device: torch.device) -> None:
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    distributed_attention, _ = _build_attention_modules()
    q, k, v = _qkv(device)

    forward_batch = ForwardBatch(data_type="dummy")
    with torch.inference_mode(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
    ):
        expected, expected_replicated = distributed_attention(q, k, v)

    torch._dynamo.reset()
    compiled_forward = torch.compile(distributed_attention.forward, backend="eager", fullgraph=True)

    with torch.inference_mode(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
    ):
        actual, actual_replicated = compiled_forward(q, k, v)

    assert expected_replicated is None
    assert actual_replicated is None
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _run_worker() -> None:
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    try:
        maybe_init_distributed_environment_and_model_parallel(1, SP_WORLD_SIZE)
        _assert_single_rank_dense_path_avoids_collectives(device)
        _assert_single_rank_dense_path_compiles_fullgraph(device)
    finally:
        cleanup_dist_env_and_memory()


def test_distributed_attention_single_rank_dense_compile_path(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")
    if torch.cuda.device_count() < SP_WORLD_SIZE:
        pytest.skip(f"This test requires at least {SP_WORLD_SIZE} CUDA device.")

    cmd = [
        "torchrun",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(SP_WORLD_SIZE),
        "--master_port",
        str(_free_port()),
        str(Path(__file__).resolve()),
        "--worker",
    ]
    env = os.environ.copy()
    env["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    process = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tmp_path)
    if process.returncode != 0:
        raise RuntimeError(
            f"distributed attention compile worker failed with code {process.returncode}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.worker:
        raise SystemExit("This module is intended to be run by pytest.")
    _run_worker()
