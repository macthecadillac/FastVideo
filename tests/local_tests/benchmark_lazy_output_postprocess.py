# SPDX-License-Identifier: Apache-2.0
"""Benchmark lazy no-return output against eager frame construction."""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from fastvideo import VideoGenerator


def _stage_time(result: dict[str, Any], stage_name: str) -> float | None:
    logging_info = result.get("logging_info")
    if logging_info is None:
        return None
    stages = getattr(logging_info, "stages", None)
    if stages is None and isinstance(logging_info, dict):
        stages = logging_info.get("stages")
    if not stages:
        return None
    stage_data = stages.get(stage_name)
    if not isinstance(stage_data, dict):
        return None
    elapsed = stage_data.get("execution_time")
    return float(elapsed) if elapsed is not None else None


def _shutdown_generator(generator: VideoGenerator | None) -> None:
    if generator is None:
        return
    executor = getattr(generator, "executor", None)
    shutdown = getattr(executor, "shutdown", None)
    if shutdown is not None:
        shutdown()


def _build_generator(args: argparse.Namespace) -> VideoGenerator:
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
        output_type="pil",
        pin_cpu_memory=True,
    )


def _run_once(
    generator: VideoGenerator,
    prompt: str,
    generation_kwargs: dict[str, Any],
    *,
    return_frames: bool,
) -> dict[str, Any]:
    kwargs = dict(generation_kwargs)
    kwargs["return_frames"] = return_frames
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = generator.generate_video(prompt, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    frames = result.get("frames")
    samples = result.get("samples")
    metrics = {
        "elapsed_s": elapsed,
        "result_e2e_latency_s": result.get("e2e_latency"),
        "postprocess_time_s": _stage_time(result, "PostDecodeFrameProcessStage"),
        "peak_memory_mb": result.get("peak_memory_mb"),
        "samples_returned": samples is not None,
        "frames_returned": frames is not None,
        "frame_count": len(frames) if frames is not None else 0,
    }
    del result
    gc.collect()
    return metrics


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [run["elapsed_s"] for run in runs]
    e2e = [run["result_e2e_latency_s"] for run in runs if run["result_e2e_latency_s"] is not None]
    postprocess = [run["postprocess_time_s"] for run in runs if run["postprocess_time_s"] is not None]
    peak_memories = [run["peak_memory_mb"] for run in runs if run["peak_memory_mb"] is not None]
    return {
        "avg_elapsed_s": mean(elapsed),
        "individual_elapsed_s": elapsed,
        "avg_result_e2e_latency_s": mean(e2e) if e2e else None,
        "individual_result_e2e_latency_s": e2e,
        "avg_postprocess_time_s": mean(postprocess) if postprocess else None,
        "individual_postprocess_time_s": postprocess,
        "max_peak_memory_mb": max(peak_memories) if peak_memories else None,
        "individual_peak_memory_mb": peak_memories,
        "samples_returned": [run["samples_returned"] for run in runs],
        "frames_returned": [run["frames_returned"] for run in runs],
        "frame_count": [run["frame_count"] for run in runs],
    }


def _ratio(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or after == 0.0:
        return None
    return before / after


def _difference(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return before - after


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
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    original_stage_logging = os.environ.get("FASTVIDEO_STAGE_LOGGING")
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    generator = None
    try:
        generation_kwargs = {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": 1024,
            "fps": 24,
            "save_video": False,
            "neg_prompt": "low quality, blurry, distorted",
            "output_path": "/tmp/fastvideo_lazy_output_benchmark/output.mp4",
        }

        generator = _build_generator(args)
        for _ in range(args.warmup):
            _run_once(generator, args.prompt, generation_kwargs, return_frames=False)
            _run_once(generator, args.prompt, generation_kwargs, return_frames=True)

        lazy_runs = []
        eager_equivalent_runs = []
        for _ in range(args.iterations):
            lazy_runs.append(_run_once(generator, args.prompt, generation_kwargs, return_frames=False))
            eager_equivalent_runs.append(_run_once(generator, args.prompt, generation_kwargs, return_frames=True))

        lazy = _summarize(lazy_runs)
        eager_equivalent = _summarize(eager_equivalent_runs)
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
            "lazy_no_return": lazy,
            "eager_equivalent_return_frames": eager_equivalent,
            "postprocess_speedup_x": _ratio(
                eager_equivalent["avg_postprocess_time_s"],
                lazy["avg_postprocess_time_s"],
            ),
            "postprocess_saved_s": _difference(
                eager_equivalent["avg_postprocess_time_s"],
                lazy["avg_postprocess_time_s"],
            ),
            "elapsed_speedup_x": eager_equivalent["avg_elapsed_s"] / lazy["avg_elapsed_s"],
            "elapsed_saved_s": eager_equivalent["avg_elapsed_s"] - lazy["avg_elapsed_s"],
            "result_e2e_speedup_x": _ratio(
                eager_equivalent["avg_result_e2e_latency_s"],
                lazy["avg_result_e2e_latency_s"],
            ),
            "result_e2e_saved_s": _difference(
                eager_equivalent["avg_result_e2e_latency_s"],
                lazy["avg_result_e2e_latency_s"],
            ),
        }
    finally:
        _shutdown_generator(generator)
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
