# SPDX-License-Identifier: Apache-2.0
"""Check that lazy no-return postprocess preserves decoded tensor output."""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from fastvideo import VideoGenerator


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


def _install_output_capture(generator: VideoGenerator, captures: list[torch.Tensor]) -> None:
    original_execute_forward = generator.executor.execute_forward

    def execute_forward_and_capture(batch, fastvideo_args):
        output_batch = original_execute_forward(batch, fastvideo_args)
        if output_batch.output is None:
            raise RuntimeError("Forward returned no output tensor.")
        captures.append(output_batch.output.detach().float().cpu())
        return output_batch

    generator.executor.execute_forward = execute_forward_and_capture


def _generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "embedded_cfg_scale": args.embedded_cfg_scale,
        "seed": args.seed,
        "fps": 24,
        "save_video": False,
        "neg_prompt": args.negative_prompt,
        "output_path": "/tmp/fastvideo_lazy_output_parity/output.mp4",
    }


def _compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    diff = (reference - candidate).abs()
    return {
        "shape": list(reference.shape),
        "dtype_reference": str(reference.dtype),
        "dtype_candidate": str(candidate.dtype),
        "exact_equal": bool(torch.equal(reference, candidate)),
        "allclose_atol_0_rtol_0": bool(torch.allclose(reference, candidate, atol=0.0, rtol=0.0)),
        "allclose_atol_1e_6_rtol_1e_6": bool(torch.allclose(reference, candidate, atol=1e-6, rtol=1e-6)),
        "allclose_atol_1e_5_rtol_1e_5": bool(torch.allclose(reference, candidate, atol=1e-5, rtol=1e-5)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "num_different": int((diff != 0).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", default="A cinematic shot of a small robot walking through a neon city.")
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--embedded-cfg-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    captures: list[torch.Tensor] = []
    generator = None
    try:
        generator = _build_generator(args)
        _install_output_capture(generator, captures)
        kwargs = _generation_kwargs(args)

        lazy_result = generator.generate_video(args.prompt, return_frames=False, **kwargs)
        if lazy_result.get("samples") is not None or lazy_result.get("frames") is not None:
            raise RuntimeError("Lazy no-return path unexpectedly returned samples or frames.")
        if len(captures) != 1:
            raise RuntimeError(f"Expected one captured lazy output, got {len(captures)}.")
        lazy_decoded = captures[-1]

        eager_result = generator.generate_video(args.prompt, return_frames=True, **kwargs)
        eager_samples = eager_result.get("samples")
        if eager_samples is None:
            raise RuntimeError("Expected samples from return_frames=True path.")
        if eager_result.get("frames") is None:
            raise RuntimeError("Expected frames from return_frames=True path.")
        if len(captures) != 2:
            raise RuntimeError(f"Expected two captured outputs, got {len(captures)}.")
        eager_captured = captures[-1]
        eager_samples = eager_samples.detach().float().cpu()

        lazy_vs_eager_capture = _compare(lazy_decoded, eager_captured)
        lazy_vs_returned_samples = _compare(lazy_decoded, eager_samples)
        result = {
            "model_path": args.model_path,
            "num_gpus": args.num_gpus,
            "sp_size": args.sp_size,
            "tp_size": args.tp_size,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "embedded_cfg_scale": args.embedded_cfg_scale,
            "seed": args.seed,
            "lazy_result_samples_returned": lazy_result.get("samples") is not None,
            "lazy_result_frames_returned": lazy_result.get("frames") is not None,
            "eager_result_samples_returned": eager_result.get("samples") is not None,
            "eager_result_frames_returned": eager_result.get("frames") is not None,
            "lazy_vs_eager_captured_output": lazy_vs_eager_capture,
            "lazy_captured_output_vs_eager_returned_samples": lazy_vs_returned_samples,
        }
    finally:
        _shutdown_generator(generator)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not result["lazy_vs_eager_captured_output"]["allclose_atol_1e_6_rtol_1e_6"]:
        raise SystemExit("Lazy output capture parity failed at atol=1e-6, rtol=1e-6.")
    if not result["lazy_captured_output_vs_eager_returned_samples"]["allclose_atol_1e_6_rtol_1e_6"]:
        raise SystemExit("Lazy output versus returned samples parity failed at atol=1e-6, rtol=1e-6.")


if __name__ == "__main__":
    main()
