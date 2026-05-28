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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _encoded_tensor(shape: tuple[int, ...], rank: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(0, int(torch.tensor(shape).prod().item()), device=device, dtype=torch.float32)
    return values.reshape(shape) + rank * 100_000


def _all_gather_same_shape(tensor: torch.Tensor) -> list[torch.Tensor]:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return gathered


def _assert_heads_to_sequence_layout(device: torch.device) -> None:
    from fastvideo.distributed import sequence_model_parallel_all_to_all_4D

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    batch, shard_seqlen, num_heads, head_dim = 2, 3, 4, 2
    shard_heads = num_heads // world_size

    input_ = _encoded_tensor((batch, shard_seqlen, num_heads, head_dim), rank, device)
    gathered = _all_gather_same_shape(input_)

    output = sequence_model_parallel_all_to_all_4D(input_, scatter_dim=2, gather_dim=1)
    expected = torch.cat(
        [
            source[:, :, rank * shard_heads:(rank + 1) * shard_heads, :]
            for source in gathered
        ],
        dim=1,
    )
    torch.testing.assert_close(output, expected)
    assert output.is_contiguous()


def _assert_sequence_to_heads_layout(device: torch.device) -> None:
    from fastvideo.distributed import sequence_model_parallel_all_to_all_4D

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    batch, shard_seqlen, shard_heads, head_dim = 2, 3, 2, 2
    seqlen = shard_seqlen * world_size

    contiguous_input = _encoded_tensor((batch, seqlen, shard_heads, head_dim), rank, device)
    noncontiguous_input = contiguous_input.transpose(1, 2).contiguous().transpose(1, 2)
    assert not noncontiguous_input.is_contiguous()

    for input_ in (contiguous_input, noncontiguous_input):
        gathered = _all_gather_same_shape(input_.contiguous())

        output = sequence_model_parallel_all_to_all_4D(input_, scatter_dim=1, gather_dim=2)
        expected = torch.cat(
            [
                source[:, rank * shard_seqlen:(rank + 1) * shard_seqlen, :, :]
                for source in gathered
            ],
            dim=2,
        )
        torch.testing.assert_close(output, expected)
        assert output.is_contiguous()


def _assert_roundtrip_and_backward(device: torch.device) -> None:
    from fastvideo.distributed import sequence_model_parallel_all_to_all_4D

    rank = dist.get_rank()
    batch, shard_seqlen, num_heads, head_dim = 1, 4, 4, 3
    input_ = _encoded_tensor((batch, shard_seqlen, num_heads, head_dim), rank, device)
    input_.requires_grad_(True)

    output = sequence_model_parallel_all_to_all_4D(input_, scatter_dim=2, gather_dim=1)
    roundtrip = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=2)
    torch.testing.assert_close(roundtrip, input_)

    roundtrip.sum().backward()
    torch.testing.assert_close(input_.grad, torch.ones_like(input_))


def _run_worker() -> None:
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    try:
        maybe_init_distributed_environment_and_model_parallel(1, SP_WORLD_SIZE)
        _assert_heads_to_sequence_layout(device)
        _assert_sequence_to_heads_layout(device)
        _assert_roundtrip_and_backward(device)
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def test_all_to_all_4d_preserves_layout(tmp_path: Path) -> None:
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
            f"all-to-all layout worker failed with code {process.returncode}\n"
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
