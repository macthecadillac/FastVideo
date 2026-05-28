# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import socket
import subprocess
from pathlib import Path

import pytest
import torch

SP_WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_async_sp_gather_and_distributed_attention_no_grad(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")
    if torch.cuda.device_count() < SP_WORLD_SIZE:
        pytest.skip(f"This test requires at least {SP_WORLD_SIZE} CUDA devices.")

    output_path = tmp_path / "async_sp_ok.pt"
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
        "--output",
        str(output_path),
    ]
    env = os.environ.copy()
    env["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"async SP worker failed with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    payload = torch.load(output_path, map_location="cpu")
    assert payload["async_gather_shape"] == (1, 3, 4, 4)
    assert payload["attention_output_shape"] == (1, 4, 4, 32)
    assert payload["replicated_output_shape"] == (1, 3, 4, 32)


def _run_worker(output_path: Path) -> None:
    import torch.distributed as dist

    from fastvideo.attention.layer import DistributedAttention
    from fastvideo.distributed import cleanup_dist_env_and_memory
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_gather
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_gather_async
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.platforms import AttentionBackendEnum

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    try:
        maybe_init_distributed_environment_and_model_parallel(1, SP_WORLD_SIZE)

        with torch.no_grad():
            local = torch.arange(1 * 3 * 2 * 4, device=device, dtype=torch.float32).reshape(1, 3, 2, 4)
            local = local + rank * 1000
            gathered_async = sequence_model_parallel_all_gather_async(local, dim=2).wait()
            gathered_sync = sequence_model_parallel_all_gather(local, dim=2)
            expected = torch.cat((local - rank * 1000, local - rank * 1000 + 1000), dim=2)
            torch.testing.assert_close(gathered_async, expected)
            torch.testing.assert_close(gathered_async, gathered_sync)

            batch_size = 1
            local_seq_len = 4
            full_seq_len = local_seq_len * SP_WORLD_SIZE
            num_heads = 4
            head_dim = 32
            replicated_seq_len = 3

            generator = torch.Generator(device="cpu")
            generator.manual_seed(2026)

            full_q = torch.randn(batch_size, full_seq_len, num_heads, head_dim, generator=generator).to(device)
            full_k = torch.randn(batch_size, full_seq_len, num_heads, head_dim, generator=generator).to(device)
            full_v = torch.randn(batch_size, full_seq_len, num_heads, head_dim, generator=generator).to(device)
            replicated_q = torch.randn(
                batch_size,
                replicated_seq_len,
                num_heads,
                head_dim,
                generator=generator,
            ).to(device)
            replicated_k = torch.randn(
                batch_size,
                replicated_seq_len,
                num_heads,
                head_dim,
                generator=generator,
            ).to(device)
            replicated_v = torch.randn(
                batch_size,
                replicated_seq_len,
                num_heads,
                head_dim,
                generator=generator,
            ).to(device)

            seq_start = rank * local_seq_len
            seq_end = seq_start + local_seq_len
            attention = DistributedAttention(
                num_heads=num_heads,
                head_size=head_dim,
                supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
            )
            forward_batch = ForwardBatch(data_type="dummy")
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch,
            ):
                attention_output, replicated_output = attention(
                    full_q[:, seq_start:seq_end],
                    full_k[:, seq_start:seq_end],
                    full_v[:, seq_start:seq_end],
                    original_seq_len=full_seq_len,
                    replicated_q=replicated_q,
                    replicated_k=replicated_k,
                    replicated_v=replicated_v,
                )

            assert replicated_output is not None
            assert torch.isfinite(attention_output).all()
            assert torch.isfinite(replicated_output).all()

        if rank == 0:
            torch.save(
                {
                    "async_gather_shape": tuple(gathered_async.shape),
                    "attention_output_shape": tuple(attention_output.shape),
                    "replicated_output_shape": tuple(replicated_output.shape),
                },
                output_path,
            )
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.worker or args.output is None:
        raise SystemExit("This module is intended to be run by pytest or with --worker --output.")
    _run_worker(Path(args.output))
