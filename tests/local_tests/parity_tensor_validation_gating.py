# SPDX-License-Identifier: Apache-2.0
"""Check output parity between default and full tensor validation modes."""

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


def _build_generator(args: argparse.Namespace, enable_full_tensor_validation: bool) -> VideoGenerator:
    return VideoGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        sp_size=args.sp_size,
        tp_size=args.tp_size,
        flow_shift=7.0,
        vae_sp=args.num_gpus > 1,
        vae_tiling=True,
        text_encoder_precisions=("fp32", ),
        enable_stage_verification=True,
        enable_full_tensor_validation=enable_full_tensor_validation,
        output_type="latent",
        pin_cpu_memory=False,
    )


def _run_mode(args: argparse.Namespace, enable_full_tensor_validation: bool) -> torch.Tensor:
    generator = None
    try:
        generator = _build_generator(args, enable_full_tensor_validation)
        result = generator.generate_video(
            args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            embedded_cfg_scale=args.embedded_cfg_scale,
            seed=args.seed,
            fps=24,
            save_video=False,
            return_frames=True,
            neg_prompt=args.negative_prompt,
            output_path="/tmp/fastvideo_validation_parity/output.mp4",
        )
        samples = result.get("samples")
        if samples is None:
            raise RuntimeError("Expected latent samples when return_frames=True and output_type='latent'.")
        return samples.detach().float().cpu()
    finally:
        _shutdown_generator(generator)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

    default_samples = _run_mode(args, enable_full_tensor_validation=False)
    full_validation_samples = _run_mode(args, enable_full_tensor_validation=True)
    comparison = _compare(default_samples, full_validation_samples)
    comparison.update({
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
        "reference_mode": "enable_full_tensor_validation=False",
        "candidate_mode": "enable_full_tensor_validation=True",
    })

    payload = json.dumps(comparison, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not comparison["allclose_atol_1e_6_rtol_1e_6"]:
        raise SystemExit("Validation gating parity failed at atol=1e-6, rtol=1e-6.")


if __name__ == "__main__":
    main()
