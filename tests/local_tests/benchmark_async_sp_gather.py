# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

CASES = [
    {
        "name": "mid_seq_text",
        "full_seq_len": 8192,
        "rep_seq_len": 256,
        "num_heads": 16,
        "head_dim": 128,
        "iters": 100,
        "warmup": 20,
    },
    {
        "name": "long_seq_text",
        "full_seq_len": 32768,
        "rep_seq_len": 256,
        "num_heads": 16,
        "head_dim": 128,
        "iters": 40,
        "warmup": 10,
    },
    {
        "name": "replicated_heavy",
        "full_seq_len": 8192,
        "rep_seq_len": 2048,
        "num_heads": 16,
        "head_dim": 128,
        "iters": 60,
        "warmup": 10,
    },
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_worker(output_path: Path, repeats: int, world_size: int) -> None:
    cmd = [
        "torchrun",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(world_size),
        "--master_port",
        str(_free_port()),
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(output_path),
        "--repeats",
        str(repeats),
        "--world-size",
        str(world_size),
    ]
    env = os.environ.copy()
    env["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    result.check_returncode()


def _summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for case in CASES:
        name = case["name"]
        sync = [item["ms_per_iter"] for item in results if item["case"] == name and item["mode"] == "sync"]
        async_ = [item["ms_per_iter"] for item in results if item["case"] == name and item["mode"] == "async"]
        sync_median = statistics.median(sync)
        async_median = statistics.median(async_)
        summary.append({
            "case": name,
            "sync_ms": sync_median,
            "async_ms": async_median,
            "delta_ms": async_median - sync_median,
            "speedup": sync_median / async_median,
            "delta_pct": (async_median / sync_median - 1.0) * 100.0,
        })
    return summary


def _print_summary(summary: list[dict[str, Any]]) -> None:
    print("\ncase,sync_ms,async_ms,delta_ms,delta_pct,speedup")
    for item in summary:
        print(
            f"{item['case']},{item['sync_ms']:.4f},{item['async_ms']:.4f},"
            f"{item['delta_ms']:.4f},{item['delta_pct']:.2f},{item['speedup']:.4f}x")


def _run_one_iteration(mode: str, main_output: torch.Tensor, replicated_output: torch.Tensor) -> None:
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_gather
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_gather_async
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_to_all_4D

    if mode == "sync":
        replicated_gathered = sequence_model_parallel_all_gather(replicated_output, dim=2)
        output = sequence_model_parallel_all_to_all_4D(main_output, scatter_dim=1, gather_dim=2)
    elif mode == "async":
        replicated_gather = sequence_model_parallel_all_gather_async(replicated_output, dim=2)
        output = sequence_model_parallel_all_to_all_4D(main_output, scatter_dim=1, gather_dim=2)
        replicated_gathered = replicated_gather.wait()
    else:
        raise ValueError(f"Unsupported mode {mode!r}")

    if output.numel() == 0 or replicated_gathered.numel() == 0:
        raise RuntimeError("Unexpected empty benchmark output")


def _measure_case(case: dict[str, Any], mode: str, device: torch.device, world_size: int) -> float:
    import torch.distributed as dist

    rank = dist.get_rank()
    heads_per_rank = case["num_heads"] // world_size
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026 + rank)
    main_output = torch.randn(
        1,
        case["full_seq_len"],
        heads_per_rank,
        case["head_dim"],
        dtype=torch.bfloat16,
        generator=generator,
    ).to(device)
    replicated_output = torch.randn(
        1,
        case["rep_seq_len"],
        heads_per_rank,
        case["head_dim"],
        dtype=torch.bfloat16,
        generator=generator,
    ).to(device)

    for _ in range(case["warmup"]):
        _run_one_iteration(mode, main_output, replicated_output)

    torch.cuda.synchronize(device)
    dist.barrier()
    start = time.perf_counter()
    for _ in range(case["iters"]):
        _run_one_iteration(mode, main_output, replicated_output)
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.perf_counter() - start

    elapsed_tensor = torch.tensor(elapsed, device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    return float(elapsed_tensor.item() * 1000.0 / case["iters"])


def _run_worker(output_path: Path, repeats: int, world_size: int) -> None:
    import torch.distributed as dist

    from fastvideo.distributed import cleanup_dist_env_and_memory
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    try:
        maybe_init_distributed_environment_and_model_parallel(1, world_size)
        results = []
        with torch.no_grad():
            for repeat in range(repeats):
                for case in CASES:
                    for mode in ("sync", "async"):
                        ms_per_iter = _measure_case(case, mode, device, world_size)
                        if rank == 0:
                            result = {
                                "repeat": repeat,
                                "case": case["name"],
                                "mode": mode,
                                "ms_per_iter": ms_per_iter,
                            }
                            print(json.dumps(result), flush=True)
                            results.append(result)
        if rank == 0:
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--world-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker:
        if args.output is None:
            raise SystemExit("--output is required in worker mode")
        _run_worker(args.output, args.repeats, args.world_size)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "async_sp_benchmark.json"
        _launch_worker(output_path, args.repeats, args.world_size)
        results = json.loads(output_path.read_text(encoding="utf-8"))
    _print_summary(_summarize(results))


if __name__ == "__main__":
    main()
