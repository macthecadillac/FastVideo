# Point 1 State: Gate Production Tensor Validation and NaN Checks

This file tracks work for `OPTIMIZATION.md` point 1 on branch
`opt-gate-validation-checks`.

## Handoff State

Last updated: 2026-05-29 after validation milestone.

Branch state:

- Current branch: `opt-gate-validation-checks`.
- Base branch/commit: `main` at
  `ba4c02d883fe01ed1bcc8143dbbcc6c4d8891d91`.
- Known unrelated workspace state remains outside this work:
  `fastvideo/tests/modal/launch_l40s_job.py` is staged from earlier workspace
  state, and several root/local artifacts remain untracked. Do not include or
  revert them unless the user explicitly asks.

Current milestone:

- Implemented and validated the point-1 gating change.

Next resume point:

- Commit and push the scoped point-1 changes on `opt-gate-validation-checks`.
- Before moving to point 2, verify the commit includes this state file and excludes unrelated workspace artifacts.

## Scope

Goal:

- Make production inference/training avoid expensive full tensor validation and
  per-step NaN scans by default.
- Keep cheap structural validation where it catches integration errors without
  synchronizing CUDA work.
- Retain opt-in diagnostics for debugging numerical issues.

Non-goals:

- Do not remove validation surfaces completely.
- Do not silently ignore shape, dtype, or missing-key contract errors.
- Do not change model numerics or pipeline outputs.

## Progress Log

### Milestone 0: Branch Setup

- Status: complete.
- Branch `opt-gate-validation-checks` was created from `main`.
- No code has been changed yet.

### Milestone 1: Implementation Complete, Validation Pending

- Status: complete.
- Added `FastVideoArgs.enable_full_tensor_validation`, `EngineConfig.enable_full_tensor_validation`,
  legacy API config compatibility, and `VideoGenerator.from_pretrained` convenience-key routing.
- Added `FASTVIDEO_FULL_TENSOR_VALIDATION` as an environment override for debug runs.
- Changed `fastvideo/pipelines/stages/validators.py` so default stage validators still check cheap
  structure/type/dimensions but skip full tensor-value NaN scans. Full scans are now enabled by either
  `enable_full_tensor_validation=True`, `FASTVIDEO_FULL_TENSOR_VALIDATION=1`, or an explicit validator
  context.
- Changed `PipelineStage.__call__` to run `verify_input` and `verify_output` inside a tensor-validation
  context, preserving existing `enable_stage_verification=True` behavior while making expensive value
  checks debug-only.
- Replaced explicit hot-path NaN assertions in denoising stages with `assert_tensor_has_no_nan(...)`.
  The affected stages are:
  - `fastvideo/pipelines/stages/denoising.py`
  - `fastvideo/pipelines/stages/sr_denoising.py`
  - `fastvideo/pipelines/stages/gamecraft_denoising.py`
  - `fastvideo/pipelines/stages/hyworld_denoising.py`
  - `fastvideo/pipelines/stages/causal_denoising.py`
  - `fastvideo/pipelines/stages/matrixgame2_denoising.py`
- Also gated the MagiHuman T5-Gemma postprocess hidden-state NaN assert through the same helper.
- Added `fastvideo/tests/stages/test_tensor_validation_gating.py` for the intended default/debug behavior.
- Added `tests/local_tests/benchmark_tensor_validation_gating.py` to quantify the isolated cost of default
  structural validation versus full NaN scans on GPU.

Handoff checkpoint:

- Current branch: `opt-gate-validation-checks`.
- Important uncommitted files in scope:
  `fastvideo/envs.py`, `fastvideo/fastvideo_args.py`, `fastvideo/api/schema.py`,
  `fastvideo/api/compat.py`, `fastvideo/entrypoints/video_generator.py`,
  `fastvideo/pipelines/stages/base.py`, `fastvideo/pipelines/stages/validators.py`,
  denoising stage files listed above, `fastvideo/pipelines/basic/magi_human/pipeline_configs.py`,
  `fastvideo/tests/api/test_parser.py`, `fastvideo/tests/stages/test_tensor_validation_gating.py`,
  `tests/local_tests/benchmark_tensor_validation_gating.py`, and this state file.
- Remaining known unrelated workspace state is still present and must be ignored unless explicitly requested:
  staged `fastvideo/tests/modal/launch_l40s_job.py`, root `.dockerignore`, `ATTN_HOT_PATH.md`,
  `OPTIMIZATION.md`, and the Flux/local test artifact directories.
- Next commands to run should include local `pytest` for the new tests, py-compile/pre-commit on changed
  non-excluded files, then Modal L40S pytest and benchmark runs using `--apply-local-patch`.

### Milestone 2: Test and Benchmark Results Recorded

- Status: complete.
- Local syntax:
  - `python -m py_compile` passed for all touched source files, the new stage test, parser test, and local
    benchmark script.
  - `git diff --check` passed.
  - `rg -n "torch\\.isnan|contains nan|contains NaN" fastvideo/pipelines/stages fastvideo/pipelines/basic/magi_human/pipeline_configs.py`
    now reports only the gated helper functions in `validators.py`.
- Local pre-commit:
  - `pre-commit run --files ...` passed for changed non-excluded files and `OPTIMIZATION_POINT_1_STATE.md`.
  - Hooks reported: `yapf`, `ruff`, `codespell`, `PyMarkdown`, GitHub Actions workflow lint skipped as no files,
    `mypy`, filename spacing, and suggestion all passed.
- Local pytest:
  - `pytest fastvideo/tests/stages/test_tensor_validation_gating.py fastvideo/tests/api/test_parser.py -q`
    could not run on the CPU-only sandbox. Importing the installed `fastvideo_kernel` package caused Triton to
    query an active CUDA driver and fail with `RuntimeError: 0 active drivers ([]). There should only be one.`
  - This is an environment limitation of the local sandbox rather than a test failure; the same tests were run on
    Modal L40S below.

Modal L40S pytest:

- Runner command:
  `UV_TOOL_DIR=/tmp/uv-tools uvx --cache-dir /tmp/uv-cache --from modal modal run fastvideo/tests/modal/launch_l40s_job.py --num-gpus 1 --install-extra none --apply-local-patch --patch-paths <point-1 paths> --command "pytest fastvideo/tests/stages/test_tensor_validation_gating.py fastvideo/tests/api/test_parser.py -q"`
- Base commit on remote: `ba4c02d883fe01ed1bcc8143dbbcc6c4d8891d91`.
- Local patch applied: yes.
- First run result: failed because the new test helper used `FastVideoArgs()` without required `model_path`.
  Fixed by changing the helper to `FastVideoArgs(model_path="test-model")`.
- Final run URL: `https://modal.com/apps/hao-ai-lab/main/ap-f33oWLRwaibzIuG3xoWgbM`.
- Final result: `12 passed in 0.05s`.
- Returned job summary:
  `{'command': 'pytest fastvideo/tests/stages/test_tensor_validation_gating.py fastvideo/tests/api/test_parser.py -q',
  'git_repo': 'https://github.com/macthecadillac/FastVideo.git',
  'git_commit': 'ba4c02d883fe01ed1bcc8143dbbcc6c4d8891d91',
  'install_extra': 'none', 'build_kernel': False, 'local_patch_applied': True, 'commit_volume': False}`.

Modal L40S benchmark:

- Runner command:
  `UV_TOOL_DIR=/tmp/uv-tools uvx --cache-dir /tmp/uv-cache --from modal modal run fastvideo/tests/modal/launch_l40s_job.py --num-gpus 1 --install-extra none --apply-local-patch --patch-paths <point-1 paths> --command "python tests/local_tests/benchmark_tensor_validation_gating.py --device cuda --numel 67108864 --iterations 100 --warmup 10"`
- Base commit on remote: `ba4c02d883fe01ed1bcc8143dbbcc6c4d8891d91`.
- Local patch applied: yes.
- Benchmark URL: `https://modal.com/apps/hao-ai-lab/main/ap-wFEFBxzBIKnmTAz9DEwTqN`.
- Device: `NVIDIA L40S`.
- Tensor: `67,108,864` elements, `torch.bfloat16`.
- Iterations: `100`; warmup: `10`.
- Results:
  - `validator_default_ms`: `0.0013850799999914898`
  - `validator_full_ms`: `0.36142692000002086`
  - `validator_saved_ms_per_call`: `0.36004184000002937`
  - `validator_speedup_x`: `260.9429924641476`
  - `assert_default_ms`: `0.0012382399999921745`
  - `assert_full_ms`: `0.36068733000000464`
  - `assert_saved_ms_per_call`: `0.35944909000001246`
  - `assert_speedup_x`: `291.29032336403617`
- Interpretation:
  - A default stage tensor validator now behaves like a cheap structural check and avoids the full CUDA scan.
  - For this isolated 128 MiB bf16 tensor, the default gated validator saves about `0.36 ms` per tensor check on
    L40S compared with the old full NaN-scan behavior.
  - Denoising hot paths also avoid per-step `torch.isnan(...).any()` / `.sum()` checks unless full tensor validation
    is explicitly enabled, so the end-to-end gain scales with the number of gated tensor checks executed per request.

Handoff checkpoint:

- Point 1 is ready to commit and push.
- Include these scoped files in the commit:
  `OPTIMIZATION_POINT_1_STATE.md`, `fastvideo/envs.py`, `fastvideo/fastvideo_args.py`,
  `fastvideo/api/schema.py`, `fastvideo/api/compat.py`, `fastvideo/entrypoints/video_generator.py`,
  `fastvideo/pipelines/stages/base.py`, `fastvideo/pipelines/stages/validators.py`,
  `fastvideo/pipelines/stages/denoising.py`, `fastvideo/pipelines/stages/sr_denoising.py`,
  `fastvideo/pipelines/stages/gamecraft_denoising.py`, `fastvideo/pipelines/stages/hyworld_denoising.py`,
  `fastvideo/pipelines/stages/causal_denoising.py`, `fastvideo/pipelines/stages/matrixgame2_denoising.py`,
  `fastvideo/pipelines/basic/magi_human/pipeline_configs.py`, `fastvideo/tests/api/test_parser.py`,
  `fastvideo/tests/stages/test_tensor_validation_gating.py`,
  `tests/local_tests/benchmark_tensor_validation_gating.py`.
- Do not include unrelated workspace artifacts:
  staged `fastvideo/tests/modal/launch_l40s_job.py`, root `.dockerignore`, `ATTN_HOT_PATH.md`,
  `OPTIMIZATION.md`, Flux output directories, or `tests/local_tests/flux2/`.
