# Point 4 State: Avoid Unnecessary CPU Output Copies and Frame Construction

This file tracks work for `OPTIMIZATION.md` point 4 on branch
`opt-lazy-output-postprocess`.

## Handoff State

Last updated: 2026-05-29 after complete Modal L40S validation and benchmark results.

Branch state:

- Current branch: `opt-lazy-output-postprocess`.
- Base branch/commit: `main` at
  `ba4c02d883fe01ed1bcc8143dbbcc6c4d8891d91`.
- Previous branches are complete and pushed:
  - Point 1: `origin/opt-gate-validation-checks`
  - Point 2: `origin/opt-batched-cfg-denoising`
- Known unrelated workspace state remains outside this work:
  staged `fastvideo/tests/modal/launch_l40s_job.py`, root `.dockerignore`,
  `ATTN_HOT_PATH.md`, `OPTIMIZATION.md`, Flux output directories, and
  `tests/local_tests/flux2/`. Do not include or revert them unless the user
  explicitly asks.

Current milestone:

- Point 4 implementation, tests, and benchmark are complete.
- Remote Modal L40S pytest and benchmark passed.
- Ready to commit and push `opt-lazy-output-postprocess`.

Next resume point:

- Commit only the point-4 scoped files:
  - `fastvideo/entrypoints/video_generator.py`
  - `fastvideo/tests/entrypoints/test_video_generator.py`
  - `tests/local_tests/benchmark_lazy_output_postprocess.py`
  - `OPTIMIZATION_POINT_4_STATE.md`
- Push branch `opt-lazy-output-postprocess` to origin.
- After point 4 is pushed, decide whether to continue with another requested
  point or stop for user review.

## Scope

Goal:

- Avoid CPU output allocation/copy and frame-list construction when the caller
  requests latent output or does not need frames/video files.
- Preserve demo-friendly behavior when `return_frames=True` or `save_video=True`.
- Keep API result shapes and keys compatible for existing callers.

Non-goals:

- Do not change denoising, VAE numerics, or scheduler behavior.
- Do not remove video saving or frame return support.
- Do not redesign the generator API beyond lazy output/postprocess decisions.

## Progress Log

### Milestone 0: Branch Setup

- Status: complete.
- Branch `opt-lazy-output-postprocess` was created from `main`.
- No point-4 code has been changed yet.

### Milestone 1: Output Path Audit

- Status: complete.
- Primary target:
  - `fastvideo/entrypoints/video_generator.py::_generate_video_impl`.
- Current behavior on `main`:
  - Pixel output preallocates a pinned CPU tensor named `samples` immediately after launching the forward thread,
    even when the caller set `return_frames=False` and `save_video=False`.
  - After forward, `output_batch.output` is copied to `samples` when shapes match, or copied through
    `output_batch.output.cpu()` on the slow path.
  - Pixel output always builds a `frames` list by rearranging `samples`, running `torchvision.utils.make_grid`,
    converting to `uint8`, and copying every frame to NumPy, even when neither returning frames nor saving video.
  - Latent output already skips pixel-shaped pinned preallocation, but still copies `output_batch.output.cpu()` into
    `samples` even when `return_frames=False`.
- Intended behavior:
  - Allocate/copy CPU `samples` only when the result will include `samples` (`return_frames=True`).
  - Build frame NumPy arrays only when they are needed for `return_frames=True` or `save_video=True`.
  - For save-only pixel output, build frames directly from `output_batch.output` without first materializing a full CPU
    `samples` tensor.
  - For latent output with `return_frames=False`, avoid the CPU copy entirely.
  - Preserve existing audio-only behavior: saving audio should not require video frame construction, but
    `return_frames=True` should still return the placeholder `samples` and an empty `frames` list.

Handoff checkpoint:

- Current branch: `opt-lazy-output-postprocess`.
- No implementation files had been edited yet except this state file at the end
  of this audit milestone.
- Unrelated workspace state remains present and should not be committed.

### Milestone 2: Lazy Output Postprocess Implementation

- Status: complete locally; remote validation pending.
- Changed `fastvideo/entrypoints/video_generator.py::_generate_single_video`.
- New output-copy behavior:
  - `samples` starts as `None`.
  - Pixel-shaped CPU preallocation is now gated by both
    `batch.return_frames` and `fastvideo_args.output_type != "latent"`.
  - The post-forward CPU copy only runs when `batch.return_frames=True`.
  - Shape-matched return-frame requests still use the preallocated buffer and
    `samples.copy_(output_batch.output)`.
  - Shape-mismatched return-frame requests still fall back to
    `output_batch.output.cpu()`, preserving previous compatibility for latent
    or unusual output shapes.
- New frame-building behavior:
  - `needs_frames` is true only for `return_frames=True` or save-to-disk pixel
    output.
  - No-save/no-return pixel output now skips `einops.rearrange`,
    `torchvision.utils.make_grid`, `uint8` conversion, and NumPy frame copies.
  - Save-only pixel output still builds a transient local `frames` list for
    `imageio`/ffmpeg, but it does not first materialize or return the full CPU
    `samples` tensor.
  - Latent output never enters the RGB frame construction path.
  - Audio-only output preserves the existing empty-frame-list behavior for
    save paths.
- Added regression tests in `fastvideo/tests/entrypoints/test_video_generator.py`:
  - `test_generate_single_video_skips_cpu_copy_and_frames_when_not_requested`
    uses an output object whose `.cpu()` raises and patches
    `torchvision.utils.make_grid` to fail, proving no CPU copy or frame
    construction is attempted for `save_video=False, return_frames=False`.
  - `test_generate_single_video_latent_output_without_return_frames_skips_cpu_copy`
    proves latent output with no returned frames does not call `.cpu()` and
    does not produce frames or a saved path.
  - `test_generate_single_video_return_frames_copies_samples_and_builds_frames`
    proves `return_frames=True` still returns CPU `samples` and frame arrays.
  - `test_generate_single_video_save_only_builds_transient_frames` proves
    `save_video=True, return_frames=False` writes with transient frames while
    leaving `result["samples"]` and `result["frames"]` as `None`.
- Local checks:
  - `python -m py_compile fastvideo/entrypoints/video_generator.py fastvideo/tests/entrypoints/test_video_generator.py`
    passed.
  - `python -m pytest fastvideo/tests/entrypoints/test_video_generator.py -q`
    failed during import before test collection because the CPU-only sandbox
    has no active Triton CUDA driver:
    `RuntimeError: 0 active drivers ([]). There should only be one.`

Handoff checkpoint:

- Current branch: `opt-lazy-output-postprocess`.
- Edited files for this point:
  - `fastvideo/entrypoints/video_generator.py`
  - `fastvideo/tests/entrypoints/test_video_generator.py`
  - `OPTIMIZATION_POINT_4_STATE.md`
- Next step is to add the benchmark script, run Modal L40S pytest and benchmark
  jobs, then update the `Final Test and Benchmark Results` section below before
  committing.

### Milestone 3: Remote Validation and Benchmark

- Status: complete.
- Added `tests/local_tests/benchmark_lazy_output_postprocess.py`.
- Benchmark shape:
  - Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
  - GPU: 2x L40S, sequence parallel size 2, tensor parallel size 1.
  - Output type: pixel (`output_type="pil"`).
  - Resolution: 256x256.
  - Frames: 17.
  - Inference steps: 4.
  - Guidance scale: 1.0 to isolate output/postprocess overhead from CFG.
  - Warmup: 1 lazy run and 1 eager-equivalent run.
  - Measured iterations: 3 interleaved lazy/eager-equivalent pairs.
- Benchmark comparison:
  - `lazy_no_return`: `save_video=False, return_frames=False`.
  - `eager_equivalent_return_frames`: `save_video=False, return_frames=True`.
  - This is not a branch-to-main A/B, but it directly measures the work point 4
    removes from the production no-return path. On `main`, the no-return pixel
    path still performed the same CPU sample materialization and frame list
    construction represented by the eager-equivalent mode.
- Modal pytest:
  - URL: `https://modal.com/apps/hao-ai-lab/main/ap-3Ha523YY2SB7s7S4QjL6F8`
  - Command:
    `python -m pytest fastvideo/tests/entrypoints/test_video_generator.py -q`
  - Result: `24 passed in 0.34s`.
- Modal real-pipeline benchmark:
  - URL: `https://modal.com/apps/hao-ai-lab/main/ap-ybVwyZIXitiZBG9IkZpJf3`
  - Command:
    `python tests/local_tests/benchmark_lazy_output_postprocess.py --num-gpus 2 --sp-size 2 --tp-size 1 --height 256 --width 256 --num-frames 17 --num-inference-steps 4 --guidance-scale 1.0 --warmup 1 --iterations 3`
  - Device: `NVIDIA L40S`.
  - Lazy no-return:
    - Average elapsed: `3.1034441906666594 s`.
    - Individual elapsed: `[3.1017871169999864, 3.1038316019999996, 3.1047138529999927]`.
    - Average result e2e latency: `3.1016933883333357 s`.
    - Individual result e2e latency: `[3.100010995000005, 3.102117597000003, 3.1029515729999986]`.
    - Average `PostDecodeFrameProcessStage`: `0.0000003670000031282446 s`.
    - Individual `PostDecodeFrameProcessStage`: `[0.00000027100000465907215, 0.00000043000000005122274, 0.00000040000000467443897]`.
    - Frames returned: `[false, false, false]`.
    - Samples returned: `[false, false, false]`.
    - Frame counts: `[0, 0, 0]`.
    - Peak memory: `8180.978515625 MB`.
  - Eager-equivalent return-frames:
    - Average elapsed: `3.113092885999999 s`.
    - Individual elapsed: `[3.114899782000009, 3.114189042999996, 3.1101898329999926]`.
    - Average result e2e latency: `3.111254529333332 s`.
    - Individual result e2e latency: `[3.1131289320000093, 3.1122979409999942, 3.108336714999993]`.
    - Average `PostDecodeFrameProcessStage`: `0.006295403333335041 s`.
    - Individual `PostDecodeFrameProcessStage`: `[0.007991601000000514, 0.006145274000004974, 0.004749334999999633]`.
    - Frames returned: `[true, true, true]`.
    - Samples returned: `[true, true, true]`.
    - Frame counts: `[17, 17, 17]`.
    - Peak memory: `8180.978515625 MB`.
  - Gains:
    - `PostDecodeFrameProcessStage` saved: `0.006295036333331913 s`.
    - `PostDecodeFrameProcessStage` speedup: `17153.68741055616x`.
    - End-to-end elapsed saved: `0.009648695333339674 s`.
    - End-to-end elapsed speedup: `1.0031090281443944x`.
    - Result e2e saved: `0.009561140999996276 s`.
    - Result e2e speedup: `1.0030825551732352x`.
  - Interpretation:
    - The postprocess-stage time is effectively removed for no-save/no-return
      pixel generation.
    - The end-to-end speedup is intentionally small for this short 4-step
      profile because text encoding, denoising, and VAE decode dominate the
      ~3.1 s runtime; the removed work is still deterministic, repeated for
      every response that does not need frames or a saved video, and scales
      with frame count, resolution, and batch size.
    - Peak GPU memory was unchanged because this optimization targets CPU
      materialization and Python/NumPy frame construction after GPU decode, not
      model activation memory.

## Final Test and Benchmark Results

- Local syntax:
  - `python -m py_compile fastvideo/entrypoints/video_generator.py fastvideo/tests/entrypoints/test_video_generator.py`
    passed.
  - `python -m py_compile tests/local_tests/benchmark_lazy_output_postprocess.py fastvideo/entrypoints/video_generator.py fastvideo/tests/entrypoints/test_video_generator.py`
    passed.
- Local lint/format:
  - `git diff --check -- fastvideo/entrypoints/video_generator.py fastvideo/tests/entrypoints/test_video_generator.py tests/local_tests/benchmark_lazy_output_postprocess.py OPTIMIZATION_POINT_4_STATE.md`
    passed.
  - `pre-commit run --files fastvideo/entrypoints/video_generator.py fastvideo/tests/entrypoints/test_video_generator.py tests/local_tests/benchmark_lazy_output_postprocess.py OPTIMIZATION_POINT_4_STATE.md`
    passed all hooks: yapf, ruff, codespell, PyMarkdown, mypy, filename spacing,
    and suggestion.
- Local pytest:
  - `python -m pytest fastvideo/tests/entrypoints/test_video_generator.py -q`
    failed during import before collection because the local CPU-only sandbox
    has no active Triton CUDA driver:
    `RuntimeError: 0 active drivers ([]). There should only be one.`
  - Modal L40S pytest above is the authoritative test result for this branch.
- Remote pytest:
  - `24 passed in 0.34s` on Modal L40S.
- Remote benchmark:
  - Lazy no-return average elapsed: `3.1034441906666594 s`.
  - Eager-equivalent return-frames average elapsed: `3.113092885999999 s`.
  - End-to-end elapsed saved: `0.009648695333339674 s`.
  - Lazy no-return average postprocess stage: `0.0000003670000031282446 s`.
  - Eager-equivalent average postprocess stage: `0.006295403333335041 s`.
  - Postprocess stage saved: `0.006295036333331913 s`.

Final handoff checkpoint:

- Branch: `opt-lazy-output-postprocess`.
- Point 4 scoped files are complete and ready for commit.
- Commit with a scoped path list or `git commit --only` because unrelated
  staged/untracked workspace files remain present.
- Do not commit the pre-existing staged `fastvideo/tests/modal/launch_l40s_job.py`
  unless the user explicitly asks.
