# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import socket
import subprocess
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

SP_WORLD_SIZE = 2
SEED = 2026


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _encoded_tensor(shape: tuple[int, ...], rank: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(0, int(torch.tensor(shape).prod().item()), device=device, dtype=torch.float32)
    return values.reshape(shape) + rank * 100_000


def _assert_async_all_gather_matches_sync(device: torch.device) -> None:
    from fastvideo.distributed import sequence_model_parallel_all_gather
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_gather_async

    local = _encoded_tensor((1, 4, 2, 8), dist.get_rank(), device)
    expected = sequence_model_parallel_all_gather(local, dim=2)
    handle = sequence_model_parallel_all_gather_async(local, dim=2)
    actual = handle.wait()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(handle.wait(), expected)


def _assert_async_all_gather_rejects_grad_enabled_input(device: torch.device) -> None:
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_gather_async

    local = _encoded_tensor((1, 2, 2, 4), dist.get_rank(), device)
    local.requires_grad_(True)
    try:
        sequence_model_parallel_all_gather_async(local, dim=2)
    except RuntimeError as error:
        assert "inference-only" in str(error)
    else:
        raise AssertionError("sequence_model_parallel_all_gather_async accepted an autograd-tracked tensor")


def _run_replicated_attention(use_async: bool, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    from fastvideo.attention import DistributedAttention
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.platforms import AttentionBackendEnum

    os.environ["FASTVIDEO_ASYNC_REPLICATED_GATHER"] = "1" if use_async else "0"
    torch.manual_seed(SEED)

    batch, seq_len_per_rank, replicated_seq_len = 1, 4, 3
    num_heads, head_dim = 4, 8
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + dist.get_rank())
    q = torch.randn(batch, seq_len_per_rank, num_heads, head_dim, device=device, generator=generator)
    k = torch.randn(batch, seq_len_per_rank, num_heads, head_dim, device=device, generator=generator)
    v = torch.randn(batch, seq_len_per_rank, num_heads, head_dim, device=device, generator=generator)
    replicated_q = torch.randn(batch, replicated_seq_len, num_heads, head_dim, device=device, generator=generator)
    replicated_k = torch.randn(batch, replicated_seq_len, num_heads, head_dim, device=device, generator=generator)
    replicated_v = torch.randn(batch, replicated_seq_len, num_heads, head_dim, device=device, generator=generator)

    attention = DistributedAttention(
        num_heads=num_heads,
        head_size=head_dim,
        causal=False,
        supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
        prefix="tests.async_replicated",
    )
    forward_batch = ForwardBatch(data_type="dummy")
    with torch.inference_mode(), set_forward_context(
        current_timestep=0,
        attn_metadata=None,
        forward_batch=forward_batch,
    ):
        output, replicated_output = attention(
            q,
            k,
            v,
            original_seq_len=seq_len_per_rank * dist.get_world_size(),
            replicated_q=replicated_q,
            replicated_k=replicated_k,
            replicated_v=replicated_v,
        )
    assert replicated_output is not None
    return output, replicated_output


def _assert_replicated_attention_async_matches_sync(device: torch.device) -> None:
    sync_output, sync_replicated_output = _run_replicated_attention(use_async=False, device=device)
    async_output, async_replicated_output = _run_replicated_attention(use_async=True, device=device)

    torch.testing.assert_close(async_output, sync_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(async_replicated_output, sync_replicated_output, rtol=1e-5, atol=1e-5)


def _run_worker() -> None:
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    try:
        maybe_init_distributed_environment_and_model_parallel(1, SP_WORLD_SIZE)
        _assert_async_all_gather_matches_sync(device)
        _assert_async_all_gather_rejects_grad_enabled_input(device)
        _assert_replicated_attention_async_matches_sync(device)
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def test_async_sequence_parallel_all_gather(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")
    if torch.cuda.device_count() < SP_WORLD_SIZE:
        pytest.skip(f"This test requires at least {SP_WORLD_SIZE} CUDA devices.")

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
            f"async all-gather worker failed with code {process.returncode}\n"
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
