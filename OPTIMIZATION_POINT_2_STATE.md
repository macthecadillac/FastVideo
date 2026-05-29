# Point 2 State: Batch CFG Instead of Running Two DiT Forwards

This file tracks work for `OPTIMIZATION.md` point 2 on branch
`opt-batched-cfg-denoising`.

## Handoff State

Last updated: 2026-05-29 after validation milestone.

Branch state:

- Current branch: `opt-batched-cfg-denoising`.
- Base branch/commit: `main` at
  `ba4c02d883fe01ed1bcc8143dbbcc6c4d8891d91`.
- Previous point-1 branch is complete and pushed at `origin/opt-gate-validation-checks`.
- Known unrelated workspace state remains outside this work:
  staged `fastvideo/tests/modal/launch_l40s_job.py`, root `.dockerignore`,
  `ATTN_HOT_PATH.md`, `OPTIMIZATION.md`, Flux output directories, and
  `tests/local_tests/flux2/`. Do not include or revert them unless the user
  explicitly asks.

Current milestone:

- Implemented and validated generic batched CFG.

Next resume point:

- Commit and push the scoped point-2 changes on `opt-batched-cfg-denoising`.
- Before moving to point 4, verify the commit includes this state file and excludes unrelated workspace artifacts.

## Scope

Goal:

- Reduce CFG inference latency by batching positive and negative transformer
  inputs into one DiT forward where the model and input kwargs support it.
- Keep a low-memory fallback for constrained GPUs or model paths that cannot
  safely batch conditional and unconditional inputs.
- Preserve output numerics within expected floating-point tolerance for the same
  model path and seed.

Non-goals:

- Do not remove the existing separate-forward CFG behavior.
- Do not change scheduler semantics, guidance-scale math, or non-CFG generation.
- Do not force batched CFG on model variants whose conditional inputs cannot be
  safely concatenated.

## Progress Log

### Milestone 0: Branch Setup

- Status: complete.
- Branch `opt-batched-cfg-denoising` was created from `main`.
- No point-2 code has been changed yet.

### Milestone 1: CFG Path Audit

- Status: complete.
- Existing batched CFG precedents:
  - `fastvideo/pipelines/stages/longcat_denoising.py`
  - `fastvideo/pipelines/stages/longcat_i2v_denoising.py`
  - `fastvideo/pipelines/stages/longcat_vc_denoising.py`
  - `fastvideo/pipelines/stages/sd35_conditioning.py`
- Primary point-2 target:
  - `fastvideo/pipelines/stages/denoising.py::DenoisingStage.forward` currently runs the transformer once for
    text-conditioned prediction, then runs it again for negative/unconditional CFG, and combines
    `noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)`.
  - This is the repeated hot path for the standard Wan/Hunyuan/LingBot-style pipelines.
- Scoped non-targets for this branch:
  - `DmdDenoisingStage` does not perform standard CFG in the audited path.
  - GameCraft/HYWorld custom stages also have separate CFG branches, but they carry specialized conditioning and
    should be treated separately after the generic stage is proven.
  - Training/distillation CFG paths are outside this inference-focused branch.
- Implementation direction:
  - Add `enable_batched_cfg` with default `True`.
  - In generic `DenoisingStage`, when CFG is enabled and `enable_batched_cfg` is true, concatenate latent inputs,
    prompt embeddings, timestep tensors, and batch-shaped conditioning kwargs along batch dimension, perform one
    transformer call, then `chunk(2)` into unconditional/text predictions.
  - Keep the existing two-forward implementation as the fallback for disabled batched CFG or CUDA OOM.
  - Avoid changing scheduler math, guidance rescale, non-CFG behavior, or model outputs beyond normal batched
    floating-point differences.

Handoff checkpoint:

- Current branch: `opt-batched-cfg-denoising`.
- No implementation files have been edited yet except this state file.
- Unrelated workspace state remains present and should not be committed.

### Milestone 2: Implementation Complete, Validation Pending

- Status: complete.
- Added `enable_batched_cfg: bool = True` to `FastVideoArgs` and `EngineConfig`.
- Added `--enable-batched-cfg` CLI support via `StoreBoolean`, so `--enable-batched-cfg false` keeps the
  lower-memory separate-forward behavior.
- Added API compatibility routing so legacy `from_pretrained`/config paths can pass `enable_batched_cfg`.
- Added `enable_batched_cfg` to `VideoGenerator.from_pretrained` convenience kwargs.
- Updated `fastvideo/tests/api/test_parser.py` for the new default.
- Implemented batched CFG in `fastvideo/pipelines/stages/denoising.py::DenoisingStage.forward`:
  - Concatenates unconditional then text-conditioned latents, prompt embeddings, timesteps, embedded guidance,
    and batch-shaped condition kwargs.
  - Performs one transformer call and splits with `chunk(2)`.
  - Reuses the existing guidance formula and guidance-rescale path.
  - Leaves non-CFG behavior unchanged.
  - Falls back to the original two-forward path when `enable_batched_cfg` is false, when inputs cannot be safely
    concatenated, or after a CUDA OOM.
- Added helper coverage in `fastvideo/tests/stages/test_batched_cfg.py`.
- Added `tests/local_tests/benchmark_batched_cfg.py`, which benchmarks the same real pipeline with
  `enable_batched_cfg=False` and `True`, extracts `DenoisingStage` time from stage logging, and reports elapsed
  time, DiT time, peak memory, speedup, and saved seconds.
- Local checks completed:
  - `python -m py_compile fastvideo/fastvideo_args.py fastvideo/api/schema.py fastvideo/api/compat.py
    fastvideo/entrypoints/video_generator.py fastvideo/pipelines/stages/denoising.py
    fastvideo/tests/stages/test_batched_cfg.py fastvideo/tests/api/test_parser.py
    tests/local_tests/benchmark_batched_cfg.py` passed.
  - `git diff --check` passed.
  - `pre-commit run --files fastvideo/fastvideo_args.py fastvideo/api/schema.py fastvideo/api/compat.py
    fastvideo/entrypoints/video_generator.py fastvideo/pipelines/stages/denoising.py
    tests/local_tests/benchmark_batched_cfg.py OPTIMIZATION_POINT_2_STATE.md` passed.

Handoff checkpoint:

- Current branch: `opt-batched-cfg-denoising`.
- Important uncommitted files in scope:
  `OPTIMIZATION_POINT_2_STATE.md`, `fastvideo/fastvideo_args.py`, `fastvideo/api/schema.py`,
  `fastvideo/api/compat.py`, `fastvideo/entrypoints/video_generator.py`,
  `fastvideo/pipelines/stages/denoising.py`, `fastvideo/tests/api/test_parser.py`,
  `fastvideo/tests/stages/test_batched_cfg.py`, and `tests/local_tests/benchmark_batched_cfg.py`.
- Known unrelated workspace state remains and must be ignored:
  staged `fastvideo/tests/modal/launch_l40s_job.py`, root `.dockerignore`, `ATTN_HOT_PATH.md`,
  `OPTIMIZATION.md`, Flux output directories, and `tests/local_tests/flux2/`.
- Next commands should use `launch_l40s_job.py` with `--apply-local-patch` over the point-2 path list.

### Milestone 3: Test and Benchmark Results Recorded

- Status: complete.
- Local syntax/static checks:
  - `python -m py_compile fastvideo/fastvideo_args.py fastvideo/api/schema.py fastvideo/api/compat.py
    fastvideo/entrypoints/video_generator.py fastvideo/pipelines/stages/denoising.py
    fastvideo/tests/stages/test_batched_cfg.py fastvideo/tests/api/test_parser.py
    tests/local_tests/benchmark_batched_cfg.py` passed.
  - `git diff --check` passed.
  - `pre-commit run --files fastvideo/fastvideo_args.py fastvideo/api/schema.py fastvideo/api/compat.py
    fastvideo/entrypoints/video_generator.py fastvideo/pipelines/stages/denoising.py
    tests/local_tests/benchmark_batched_cfg.py OPTIMIZATION_POINT_2_STATE.md` passed after YAPF reformatted touched files
    and the nested denoising helper was updated to bind loop variables explicitly.
- Local pytest:
  - Not used as the authoritative result because this CPU-only sandbox cannot import the installed Triton-backed
    `fastvideo_kernel` package without an active CUDA driver. Modal L40S results below are the authoritative tests.

Modal L40S pytest:

- Runner command:
  `UV_TOOL_DIR=/tmp/uv-tools uvx --cache-dir /tmp/uv-cache --from modal modal run fastvideo/tests/modal/launch_l40s_job.py --num-gpus 1 --install-extra none --apply-local-patch --patch-paths <point-2 paths> --command "pytest fastvideo/tests/stages/test_batched_cfg.py fastvideo/tests/api/test_parser.py -q"`
- Base commit on remote: `ba4c02d883fe01ed1bcc8143dbbcc6c4d8891d91`.
- Local patch applied: yes.
- Final run URL: `https://modal.com/apps/hao-ai-lab/main/ap-gVHImrGzLquzwW98zc05bF`.
- Result: `12 passed in 0.06s`.
- Returned job summary:
  `{'command': 'pytest fastvideo/tests/stages/test_batched_cfg.py fastvideo/tests/api/test_parser.py -q',
  'git_repo': 'https://github.com/macthecadillac/FastVideo.git',
  'git_commit': 'ba4c02d883fe01ed1bcc8143dbbcc6c4d8891d91',
  'install_extra': 'none', 'build_kernel': False, 'local_patch_applied': True, 'commit_volume': False}`.

Modal 2x L40S real-pipeline benchmark:

- Runner command:
  `UV_TOOL_DIR=/tmp/uv-tools uvx --cache-dir /tmp/uv-cache --from modal modal run fastvideo/tests/modal/launch_l40s_job.py --num-gpus 2 --install-extra none --apply-local-patch --patch-paths <point-2 paths> --command "python tests/local_tests/benchmark_batched_cfg.py --num-gpus 2 --sp-size 2 --tp-size 1 --height 256 --width 256 --num-frames 17 --num-inference-steps 4 --warmup 1 --iterations 3"`
- Base commit on remote: `ba4c02d883fe01ed1bcc8143dbbcc6c4d8891d91`.
- Local patch applied: yes.
- Benchmark URL: `https://modal.com/apps/hao-ai-lab/main/ap-fzfwlDfV0KEaqc2AC9ZKnh`.
- Device: `NVIDIA L40S` with `num_gpus=2`, `sp_size=2`, `tp_size=1`.
- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
- Shape/profile: `height=256`, `width=256`, `num_frames=17`, `num_inference_steps=4`, `guidance_scale=3.0`,
  latent output, `save_video=False`, `return_frames=False`.
- Warmup: `1` run per mode.
- Measurement iterations: `3` runs per mode.
- Separate-forward CFG (`enable_batched_cfg=False`):
  - `avg_dit_time_s`: `1.0428565246666797`
  - `individual_dit_time_s`: `[1.0402555660000132, 1.0492859740000142, 1.0390280340000118]`
  - `avg_elapsed_s`: `3.0741438060000044`
  - `individual_elapsed_s`: `[3.0716882709999993, 3.079961042000008, 3.0707821050000064]`
  - `max_peak_memory_mb`: `8184.2294921875`
- Batched CFG (`enable_batched_cfg=True`):
  - `avg_dit_time_s`: `0.5739406980000012`
  - `individual_dit_time_s`: `[0.5663130880000153, 0.5646435849999989, 0.5908654209999895]`
  - `avg_elapsed_s`: `2.602722532666661`
  - `individual_elapsed_s`: `[2.5946601320000013, 2.5931245099999956, 2.620382955999986]`
  - `max_peak_memory_mb`: `8184.2294921875`
- Computed gains:
  - `dit_speedup_x`: `1.8170109356257524`
  - `dit_saved_s`: `0.4689158266666784`
  - `elapsed_speedup_x`: `1.181126212040106`
  - `elapsed_saved_s`: `0.4714212733333434`
- Interpretation:
  - The generic denoising DiT hot path is about `1.82x` faster for this CFG-heavy 4-step Wan 1.3B profile.
  - End-to-end latency improves by about `0.47s` (`18.1%`) despite text encoding remaining unchanged and dominating
    a large portion of this small latent-output benchmark.
  - Peak memory as reported by the pipeline was unchanged at `8184.23 MB`; this benchmark shape was small enough that
    batched activations did not increase the reported peak beyond the existing model/runtime peak.

Superseded benchmark note:

- An earlier benchmark run at `https://modal.com/apps/hao-ai-lab/main/ap-0b0bFdWRcdGywhGgog3gAd` reused a single
  multiprocess generator and toggled `generator.fastvideo_args.enable_batched_cfg` between runs. That result is invalid
  for point-2 comparison because `GpuWorker.execute_forward` uses the worker's initialization-time `self.fastvideo_args`,
  not the per-call argument object. The benchmark harness was fixed to instantiate separate generators per mode before
  the final benchmark above.

Handoff checkpoint:

- Point 2 is ready to commit and push.
- Include these scoped files in the commit:
  `OPTIMIZATION_POINT_2_STATE.md`, `fastvideo/fastvideo_args.py`, `fastvideo/api/schema.py`,
  `fastvideo/api/compat.py`, `fastvideo/entrypoints/video_generator.py`,
  `fastvideo/pipelines/stages/denoising.py`, `fastvideo/tests/api/test_parser.py`,
  `fastvideo/tests/stages/test_batched_cfg.py`, and `tests/local_tests/benchmark_batched_cfg.py`.
- Do not include unrelated workspace artifacts:
  staged `fastvideo/tests/modal/launch_l40s_job.py`, root `.dockerignore`, `ATTN_HOT_PATH.md`,
  `OPTIMIZATION.md`, Flux output directories, or `tests/local_tests/flux2/`.
