# SPDX-License-Identifier: Apache-2.0
"""Benchmark separate-forward CFG against batched CFG for a real pipeline."""

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from fastvideo import VideoGenerator


def _stage_time(result: dict[str, Any], stage_class: str) -> float | None:
    logging_info = result.get("logging_info")
    if logging_info is None:
        return None
    stages = getattr(logging_info, "stages", None)
    if stages is None and isinstance(logging_info, dict):
        stages = logging_info.get("stages")
    if not stages:
        return None

    total = 0.0
    found = False
    for stage_name, stage_data in stages.items():
        data = stage_data if isinstance(stage_data, dict) else {}
        if stage_name == stage_class or data.get("stage_class") == stage_class:
            elapsed = data.get("execution_time")
            if elapsed is not None:
                total += float(elapsed)
                found = True
    return total if found else None


def _shutdown_generator(generator: VideoGenerator | None) -> None:
    if generator is None:
        return
    executor = getattr(generator, "executor", None)
    shutdown = getattr(executor, "shutdown", None)
    if shutdown is not None:
        shutdown()


def _run_once(generator: VideoGenerator, enabled: bool, prompt: str, generation_kwargs: dict[str, Any]) -> dict[str, Any]:
    if generator.fastvideo_args.enable_batched_cfg != enabled:
        raise ValueError("Generator was initialized with the wrong enable_batched_cfg value")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = generator.generate_video(prompt, **generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "elapsed_s": elapsed,
        "dit_time_s": _stage_time(result, "DenoisingStage"),
        "peak_memory_mb": result.get("peak_memory_mb"),
    }


def _build_generator(args: argparse.Namespace, enabled: bool) -> VideoGenerator:
    return VideoGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        sp_size=args.sp_size,
        tp_size=args.tp_size,
        flow_shift=7.0,
        vae_sp=args.num_gpus > 1,
        vae_tiling=True,
        text_encoder_precisions=("fp32", ),
        enable_stage_verification=False,
        enable_batched_cfg=enabled,
        output_type="latent",
    )


def _run_mode(args: argparse.Namespace, enabled: bool, generation_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    generator = None
    try:
        generator = _build_generator(args, enabled)
        for _ in range(args.warmup):
            _run_once(generator, enabled, args.prompt, generation_kwargs)
        return [_run_once(generator, enabled, args.prompt, generation_kwargs) for _ in range(args.iterations)]
    finally:
        _shutdown_generator(generator)


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [run["elapsed_s"] for run in runs]
    dit_times = [run["dit_time_s"] for run in runs if run["dit_time_s"] is not None]
    peak_memories = [run["peak_memory_mb"] for run in runs if run["peak_memory_mb"] is not None]
    return {
        "avg_elapsed_s": mean(elapsed),
        "individual_elapsed_s": elapsed,
        "avg_dit_time_s": mean(dit_times) if dit_times else None,
        "individual_dit_time_s": dit_times,
        "max_peak_memory_mb": max(peak_memories) if peak_memories else None,
        "individual_peak_memory_mb": peak_memories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", default="A cinematic shot of a small robot walking through a neon city.")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--sp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--embedded-cfg-scale", type=float, default=6.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    original_stage_logging = os.environ.get("FASTVIDEO_STAGE_LOGGING")
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    try:
        generation_kwargs = {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "embedded_cfg_scale": args.embedded_cfg_scale,
            "seed": 1024,
            "fps": 24,
            "save_video": False,
            "return_frames": False,
            "neg_prompt": "low quality, blurry, distorted",
            "output_path": "/tmp/fastvideo_batched_cfg_benchmark",
        }

        separate_runs = _run_mode(args, False, generation_kwargs)
        batched_runs = _run_mode(args, True, generation_kwargs)
        separate = _summarize(separate_runs)
        batched = _summarize(batched_runs)
        separate_dit = separate["avg_dit_time_s"]
        batched_dit = batched["avg_dit_time_s"]
        result = {
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "model_path": args.model_path,
            "num_gpus": args.num_gpus,
            "sp_size": args.sp_size,
            "tp_size": args.tp_size,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "separate_cfg": separate,
            "batched_cfg": batched,
            "dit_speedup_x": separate_dit / batched_dit if separate_dit and batched_dit else None,
            "dit_saved_s": separate_dit - batched_dit if separate_dit and batched_dit else None,
            "elapsed_speedup_x": separate["avg_elapsed_s"] / batched["avg_elapsed_s"],
            "elapsed_saved_s": separate["avg_elapsed_s"] - batched["avg_elapsed_s"],
        }
    finally:
        if original_stage_logging is None:
            os.environ.pop("FASTVIDEO_STAGE_LOGGING", None)
        else:
            os.environ["FASTVIDEO_STAGE_LOGGING"] = original_stage_logging

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
