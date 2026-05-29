# SPDX-License-Identifier: Apache-2.0
"""Benchmark gated tensor-value validation overhead.

This benchmark isolates the point-1 hot-path change: structural stage validators
now avoid full tensor scans unless full tensor validation is enabled.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import assert_tensor_has_no_nan, tensor_validation_context


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark(fn, device: torch.device, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--numel", type=int, default=67_108_864)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    value = torch.zeros(args.numel, device=device, dtype=torch.bfloat16)

    with tensor_validation_context(False):
        validator_default_ms = _benchmark(lambda: V.is_tensor(value), device, args.warmup, args.iterations)
        assert_default_ms = _benchmark(lambda: assert_tensor_has_no_nan(value, "value"), device, args.warmup,
                                       args.iterations)

    with tensor_validation_context(True):
        validator_full_ms = _benchmark(lambda: V.is_tensor(value), device, args.warmup, args.iterations)
        assert_full_ms = _benchmark(lambda: assert_tensor_has_no_nan(value, "value"), device, args.warmup,
                                    args.iterations)

    result = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "numel": args.numel,
        "dtype": str(value.dtype),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "validator_default_ms": validator_default_ms,
        "validator_full_ms": validator_full_ms,
        "validator_speedup_x": validator_full_ms / validator_default_ms if validator_default_ms > 0 else None,
        "validator_saved_ms_per_call": validator_full_ms - validator_default_ms,
        "assert_default_ms": assert_default_ms,
        "assert_full_ms": assert_full_ms,
        "assert_speedup_x": assert_full_ms / assert_default_ms if assert_default_ms > 0 else None,
        "assert_saved_ms_per_call": assert_full_ms - assert_default_ms,
    }

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
