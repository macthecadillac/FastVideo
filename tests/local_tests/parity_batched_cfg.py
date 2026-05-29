# SPDX-License-Identifier: Apache-2.0
"""Compare separate-forward CFG and batched CFG latent outputs."""

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


def _build_generator(args: argparse.Namespace, enable_batched_cfg: bool) -> VideoGenerator:
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
        enable_batched_cfg=enable_batched_cfg,
        output_type="latent",
        pin_cpu_memory=False,
    )


def _run_mode(args: argparse.Namespace, enable_batched_cfg: bool) -> torch.Tensor:
    generator = None
    try:
        generator = _build_generator(args, enable_batched_cfg)
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
            output_path="/tmp/fastvideo_batched_cfg_parity/output.mp4",
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
    reference_abs = reference.abs()
    relative = diff / reference_abs.clamp_min(1e-6)
    return {
        "shape": list(reference.shape),
        "dtype_reference": str(reference.dtype),
        "dtype_candidate": str(candidate.dtype),
        "exact_equal": bool(torch.equal(reference, candidate)),
        "allclose_atol_0_rtol_0": bool(torch.allclose(reference, candidate, atol=0.0, rtol=0.0)),
        "allclose_atol_1e_5_rtol_1e_5": bool(torch.allclose(reference, candidate, atol=1e-5, rtol=1e-5)),
        "allclose_atol_1e_4_rtol_1e_4": bool(torch.allclose(reference, candidate, atol=1e-4, rtol=1e-4)),
        "allclose_atol_1e_3_rtol_1e_3": bool(torch.allclose(reference, candidate, atol=1e-3, rtol=1e-3)),
        "allclose_atol_1e_2_rtol_1e_2": bool(torch.allclose(reference, candidate, atol=1e-2, rtol=1e-2)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_relative": float(relative.max().item()),
        "mean_relative": float(relative.mean().item()),
        "num_different": int((diff != 0).sum().item()),
    }


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def _mode_name(enable_batched_cfg: bool) -> str:
    return f"enable_batched_cfg={enable_batched_cfg}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", default="A cinematic shot of a small robot walking through a neon city.")
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--sp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--embedded-cfg-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--reference-batched-cfg", type=_parse_bool, default=False)
    parser.add_argument("--candidate-batched-cfg", type=_parse_bool, default=True)
    parser.add_argument("--determinism-control", action="store_true")
    parser.add_argument("--fail-on-threshold", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    reference_samples = _run_mode(args, enable_batched_cfg=args.reference_batched_cfg)
    candidate_samples = _run_mode(args, enable_batched_cfg=args.candidate_batched_cfg)
    comparison = _compare(reference_samples, candidate_samples)
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
        "reference_mode": _mode_name(args.reference_batched_cfg),
        "candidate_mode": _mode_name(args.candidate_batched_cfg),
    })
    result = {
        "comparison": comparison,
    }

    if args.determinism_control:
        reference_repeat_samples = _run_mode(args, enable_batched_cfg=args.reference_batched_cfg)
        result["reference_repeat_comparison"] = _compare(reference_samples, reference_repeat_samples)

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.fail_on_threshold and not comparison["allclose_atol_1e_2_rtol_1e_2"]:
        raise SystemExit("Batched CFG parity failed at atol=1e-2, rtol=1e-2.")


if __name__ == "__main__":
    main()
