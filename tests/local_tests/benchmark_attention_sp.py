# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark attention sequence-parallel hot paths.

Run with torchrun, for example:

    torchrun --nproc_per_node=2 tests/local_tests/benchmark_attention_sp.py \
        --output /tmp/attention_sp_baseline.json

The benchmark intentionally uses synthetic tensors and real FastVideo
distributed/attention helpers. It does not require model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _parse_csv_ints(value: str, *, expected: int, name: str) -> tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != expected:
        raise ValueError(f"{name} must have {expected} comma-separated ints, got {value!r}")
    return parts


def _dtype_from_string(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _ensure_torchrun_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def _rank_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def _memory_stats() -> dict[str, int]:
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _time_cuda_ms(
    fn: Callable[[], torch.Tensor | Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    last_output = None
    for _ in range(iterations):
        last_output = fn()
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / iterations
    result: dict[str, Any] = {
        "avg_ms": avg_ms,
        "iterations": iterations,
        **_memory_stats(),
    }
    if torch.is_tensor(last_output):
        result["output_shape"] = list(last_output.shape)
        result["output_is_contiguous"] = bool(last_output.is_contiguous())
        result["output_stride"] = list(last_output.stride())
    return result


def _time_cpu_us(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    last_output = None
    for _ in range(iterations):
        last_output = fn()
    elapsed_us = (time.perf_counter() - start) * 1_000_000.0 / iterations
    result: dict[str, Any] = {
        "avg_us": elapsed_us,
        "iterations": iterations,
    }
    if last_output is not None:
        result["metadata_type"] = type(last_output).__name__
    return result


def _build_tensors(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    from fastvideo.distributed import get_sp_parallel_rank, get_sp_world_size

    world_size = get_sp_world_size()
    heads = args.num_heads
    if heads % world_size != 0:
        raise ValueError(f"--num-heads ({heads}) must be divisible by SP world size ({world_size})")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + get_sp_parallel_rank())
    batch = args.batch_size
    seq = args.seq_len_per_rank
    head_dim = args.head_dim
    heads_per_rank = heads // world_size

    q = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype, generator=generator)
    k = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype, generator=generator)
    v = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype, generator=generator)
    qkv = torch.cat([q, k, v], dim=0)
    seq_to_heads = torch.randn(
        batch,
        seq * world_size,
        heads_per_rank,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    replicated = torch.randn(
        batch,
        args.replicated_seq_len,
        heads_per_rank,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    replicated_q = torch.randn(
        batch,
        args.replicated_seq_len,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    replicated_k = torch.randn(
        batch,
        args.replicated_seq_len,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    replicated_v = torch.randn(
        batch,
        args.replicated_seq_len,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "qkv": qkv,
        "seq_to_heads": seq_to_heads,
        "replicated": replicated,
        "replicated_q": replicated_q,
        "replicated_k": replicated_k,
        "replicated_v": replicated_v,
    }


def _benchmark_communication(
    args: argparse.Namespace,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> list[dict[str, Any]]:
    from fastvideo.distributed import sequence_model_parallel_all_gather, sequence_model_parallel_all_to_all_4D

    del device
    cases: list[dict[str, Any]] = []

    qkv = tensors["qkv"]
    seq_to_heads = tensors["seq_to_heads"]
    replicated = tensors["replicated"]

    roundtrip = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
    roundtrip = sequence_model_parallel_all_to_all_4D(roundtrip, scatter_dim=1, gather_dim=2)
    roundtrip_max_abs_diff = float((roundtrip - qkv).abs().max().item())

    cases.append({
        "name": "sp_all_to_all_heads_to_sequence",
        "estimated_collectives": 1,
        "input_shape": list(qkv.shape),
        "roundtrip_max_abs_diff": roundtrip_max_abs_diff,
        **_time_cuda_ms(
            lambda: sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    })
    cases.append({
        "name": "sp_all_to_all_sequence_to_heads",
        "estimated_collectives": 1,
        "input_shape": list(seq_to_heads.shape),
        **_time_cuda_ms(
            lambda: sequence_model_parallel_all_to_all_4D(seq_to_heads, scatter_dim=1, gather_dim=2),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    })
    cases.append({
        "name": "sp_all_gather_replicated_tokens",
        "estimated_collectives": 1,
        "input_shape": list(replicated.shape),
        **_time_cuda_ms(
            lambda: sequence_model_parallel_all_gather(replicated, dim=2),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    })
    return cases


def _benchmark_distributed_attention(
    args: argparse.Namespace,
    tensors: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    from fastvideo.attention import DistributedAttention
    from fastvideo.distributed import get_sp_world_size
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.platforms import AttentionBackendEnum
    from fastvideo.utils import set_mixed_precision_policy

    set_mixed_precision_policy(param_dtype=dtype, reduce_dtype=dtype, output_dtype=dtype)
    backend = AttentionBackendEnum[args.attention_backend]
    attention = DistributedAttention(
        num_heads=args.num_heads,
        head_size=args.head_dim,
        causal=False,
        supported_attention_backends=(
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        ),
        prefix="benchmark.blocks.0.attn1",
    )
    if attention.backend != backend:
        raise RuntimeError(f"Requested {backend.name}, but DistributedAttention selected {attention.backend}")

    original_seq_len = args.seq_len_per_rank * get_sp_world_size()
    forward_batch = ForwardBatch(data_type="benchmark")

    def run_attention() -> torch.Tensor:
        with torch.inference_mode(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
        ):
            output, _ = attention(
                tensors["q"],
                tensors["k"],
                tensors["v"],
                original_seq_len=original_seq_len,
            )
        return output

    cases = [{
        "name": "distributed_attention_dense",
        "estimated_collectives": 2 if get_sp_world_size() > 1 else 0,
        "input_shape": list(tensors["q"].shape),
        "selected_backend": attention.backend.name if attention.backend is not None else None,
        **_time_cuda_ms(run_attention, warmup=args.warmup, iterations=args.iterations),
    }]

    def run_replicated_attention() -> torch.Tensor:
        with torch.inference_mode(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
        ):
            output, replicated_output = attention(
                tensors["q"],
                tensors["k"],
                tensors["v"],
                original_seq_len=original_seq_len,
                replicated_q=tensors["replicated_q"],
                replicated_k=tensors["replicated_k"],
                replicated_v=tensors["replicated_v"],
            )
        assert replicated_output is not None
        return output

    if args.include_replicated_attention:
        cases.append({
            "name": "distributed_attention_replicated",
            "estimated_collectives": 3 if get_sp_world_size() > 1 else 0,
            "input_shape": list(tensors["q"].shape),
            "replicated_input_shape": list(tensors["replicated_q"].shape),
            "async_replicated_gather": os.environ.get("FASTVIDEO_ASYNC_REPLICATED_GATHER", "0") != "0",
            "selected_backend": attention.backend.name if attention.backend is not None else None,
            **_time_cuda_ms(run_replicated_attention, warmup=args.warmup, iterations=args.iterations),
        })
    return cases


def _benchmark_metadata(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    from fastvideo.attention.backends.bsa_attn import BSAAttentionMetadataBuilder
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionMetadataBuilder,
    )

    raw_latent_shape = _parse_csv_ints(args.metadata_raw_latent_shape, expected=3, name="metadata raw latent shape")
    patch_size = _parse_csv_ints(args.metadata_patch_size, expected=3, name="metadata patch size")
    cases: list[dict[str, Any]] = []

    vsa_builder = VideoSparseAttentionMetadataBuilder()
    cases.append({
        "name": "metadata_vsa",
        "raw_latent_shape": list(raw_latent_shape),
        "patch_size": list(patch_size),
        **_time_cpu_us(
            lambda: vsa_builder.build(
                current_timestep=0,
                raw_latent_shape=raw_latent_shape,
                patch_size=patch_size,
                VSA_sparsity=args.vsa_sparsity,
                device=device,
            ),
            warmup=args.metadata_warmup,
            iterations=args.metadata_iterations,
        ),
    })

    bsa_builder = BSAAttentionMetadataBuilder()
    cases.append({
        "name": "metadata_bsa",
        "raw_latent_shape": list(raw_latent_shape),
        "patch_size": list(patch_size),
        **_time_cpu_us(
            lambda: bsa_builder.build(
                current_timestep=0,
                raw_latent_shape=raw_latent_shape,
                patch_size=patch_size,
                device=device,
            ),
            warmup=args.metadata_warmup,
            iterations=args.metadata_iterations,
        ),
    })

    try:
        from fastvideo.attention.backends.vmoba import VideoMobaAttentionMetadataBuilder
    except ImportError as error:
        cases.append({
            "name": "metadata_vmoba",
            "skipped": True,
            "reason": str(error),
        })
    else:
        vmoba_builder = VideoMobaAttentionMetadataBuilder()
        cases.append({
            "name": "metadata_vmoba",
            "raw_latent_shape": list(raw_latent_shape),
            "patch_size": list(patch_size),
            **_time_cpu_us(
                lambda: vmoba_builder.build(
                    current_timestep=0,
                    raw_latent_shape=raw_latent_shape,
                    patch_size=patch_size,
                    temporal_chunk_size=4,
                    temporal_topk=2,
                    spatial_chunk_size=(4, 4),
                    spatial_topk=4,
                    st_chunk_size=(4, 4, 4),
                    st_topk=4,
                    device=device,
                ),
                warmup=args.metadata_warmup,
                iterations=args.metadata_iterations,
            ),
        })

    return cases


def _summarize_rank_cases(gathered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for rank_result in gathered:
        for case in rank_result["cases"]:
            by_name.setdefault(case["name"], []).append(case)

    summary = []
    for name, cases in sorted(by_name.items()):
        item: dict[str, Any] = {
            "name": name,
            "num_ranks": len(cases),
        }
        if all("avg_ms" in case for case in cases):
            item["max_rank_avg_ms"] = max(case["avg_ms"] for case in cases)
            item["min_rank_avg_ms"] = min(case["avg_ms"] for case in cases)
            item["max_rank_memory_allocated_bytes"] = max(
                case["max_memory_allocated_bytes"] for case in cases
            )
        if all("avg_us" in case for case in cases):
            item["max_rank_avg_us"] = max(case["avg_us"] for case in cases)
            item["min_rank_avg_us"] = min(case["avg_us"] for case in cases)
        if all(case.get("skipped") for case in cases):
            item["skipped"] = True
            item["reason"] = cases[0].get("reason")
        summary.append(item)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", default="communication,attention,metadata")
    parser.add_argument("--output", default="/tmp/attention_sp_benchmark.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len-per-rank", type=int, default=2048)
    parser.add_argument("--replicated-seq-len", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attention-backend", default="FLASH_ATTN", choices=["FLASH_ATTN", "TORCH_SDPA"])
    parser.add_argument("--include-replicated-attention", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--metadata-warmup", type=int, default=5)
    parser.add_argument("--metadata-iterations", type=int, default=100)
    parser.add_argument("--metadata-raw-latent-shape", default="16,64,64")
    parser.add_argument("--metadata-patch-size", default="1,2,2")
    parser.add_argument("--vsa-sparsity", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_torchrun_env()
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.attention_backend

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for attention SP benchmarking")

    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        get_sp_parallel_rank,
        get_sp_world_size,
        maybe_init_distributed_environment_and_model_parallel,
    )

    device = _rank_device()
    dtype = _dtype_from_string(args.dtype)
    sp_size = int(os.environ.get("WORLD_SIZE", "1"))
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=sp_size)
    dist.barrier()

    try:
        tensors = _build_tensors(args, device, dtype)
        suites = {suite.strip() for suite in args.suites.split(",") if suite.strip()}
        cases: list[dict[str, Any]] = []
        if "communication" in suites:
            cases.extend(_benchmark_communication(args, tensors, device))
        if "attention" in suites:
            cases.extend(_benchmark_distributed_attention(args, tensors, dtype))
        if "metadata" in suites:
            cases.extend(_benchmark_metadata(args, device))

        rank_result = {
            "rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "device": torch.cuda.get_device_name(device),
            "sp_world_size": get_sp_world_size(),
            "sp_rank": get_sp_parallel_rank(),
            "cases": cases,
        }

        gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, rank_result)

        if dist.get_rank() == 0:
            output = {
                "benchmark": "attention_sp",
                "args": vars(args),
                "world_size": dist.get_world_size(),
                "summary": _summarize_rank_cases([item for item in gathered if item is not None]),
                "ranks": gathered,
            }
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(output["summary"], indent=2, sort_keys=True), flush=True)
            print(f"Wrote benchmark results to {output_path}", flush=True)
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
