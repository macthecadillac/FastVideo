# Issue 775 TDM Handoff

Compacted: 2026-07-01
Last updated: 2026-07-10 Wan-Syn true batch-4 DGX smoke completed

This file intentionally replaces the earlier long chronological log. Older
per-run details are preserved in branch commits and Modal artifact paths; this
handoff keeps only state needed to continue work.

## 2026-07-09 Reviewer-Guided Objective/Schedule Patch

- Resumed in `/tmp/fastvideo-worktrees/issue-775-tdm` on branch
  `issue/775-tdm`, initially clean at pushed commit
  `7c6ddd8ee`. The user supplied an independent review identifying three
  likely quality culprits and asked to try those solutions and train again.
- Refreshed GitHub state with `gh` as `macthecadillac` before committing:
  issue #775 remains open, assigned to `macthecadillac`, with no new issue
  comments beyond the existing collaborator/stale-bot comments. Targeted open
  PR search for `775 OR TDM` returned `[]`; no PR draft status was changed.
- Reviewer culprits evaluated and implemented:
  - **Flow schedule mismatch**: `pipeline.flow_shift: 8` in the TDM config was
    not reaching Wan role schedulers because `WanModel` defaulted to
    `flow_shift=3.0` unless each role set it explicitly. Patched
    `fastvideo/train/models/wan/wan.py` so an omitted role `flow_shift`
    inherits `training_config.pipeline_config.flow_shift`, with explicit role
    kwargs still taking precedence and a final Wan fallback of `3.0`.
    `WanCausalModel` now passes through the same `None` default. Also changed
    `_init_timestep_mechanics()` to mirror the actual scheduler shift instead
    of separately reading the pipeline config.
  - **Validation scheduler override**: `DmdDenoisingStage` previously replaced
    the scheduler it was passed with a new `FlowMatchEulerDiscreteScheduler`
    hardcoded to `shift=8.0`. Patched it to preserve the configured scheduler,
    so validation follows the pipeline scheduler construction path instead of a
    hidden stage-local override.
  - **Terminal-only generator gradients**: `TDMMethod.single_train_step()` now
    rolls the student trajectory without gradients, then
    `_tdm_generator_loss(...)` samples a non-terminal configured TDM trajectory
    point, recomputes that student prediction with gradients, and evaluates
    teacher/critic targets at the same trajectory state. The helper now logs
    `tdm/generator/trajectory_index` and keeps `dmd_latent_vis_dict`
    `generator_timestep` aligned with the sampled generator timestep. The
    rollout helper now enforces no-grad internally; the only student-gradient
    path is the sampled recomputation in generator loss.
  - **Stochastic default path**: the Wan TDM LoRA YAML now sets
    `method.student_sample_type: ode` and `pipeline.dmd_sample_type: ode`.
    Added `PipelineConfig.dmd_sample_type` plus CLI parsing for
    `--pipeline.dmd-sample-type`, defaulting to existing `sde` behavior for
    other configs. In ODE mode, `DmdDenoisingStage` carries the effective flow
    noise from the current state to the next configured timestep instead of
    drawing new noise between steps.
- Tests/config coverage updated:
  - `tests/local_tests/tdm/test_tdm_method_unit.py` now asserts exactly one
    gradient-enabled student VSA prediction happens for generator loss and
    that it occurs at trajectory timestep `750` or `500`, not the terminal
    `250` final point.
  - `tests/local_tests/tdm/test_tdm_config_smoke.py` now asserts the TDM config
    uses ODE rollout/validation, that a Wan role built from the config inherits
    `pipeline.flow_shift == 8`, and that `DmdDenoisingStage` preserves a
    caller-provided scheduler and shift.
  - `examples/train/configs/example.yaml` now documents the role-level
    `flow_shift` default as inheriting `training.pipeline.flow_shift` when
    available, with Wan fallback `3.0`.
- Local non-test checks completed:
  - `git diff --check`: passed.
  - `python -m py_compile` on all changed Python files: passed.
  - Changed-file `pre-commit run --files ...` from the issue worktree passed
    `yapf`, `ruff`, and `codespell`, but the mypy hook failed before checking
    files with the known realpath issue `issue-775-tdm is not a valid Python
    package name`.
  - Re-ran the same changed-file pre-commit command in a temporary validation
    clone at `/tmp/fastvideo_worktrees/issue_775_tdm_clone` with the exact
    same diff applied. All hooks passed, including mypy.
- Modal validation completed before commit:
  - First targeted TDM local-test app `ap-s9sdm73lczyzjd1pxHJt4u` exposed a
    test-fixture bug: `DmdDenoisingStage` construction needs transformer
    metadata, and the smoke test used a bare `torch.nn.Identity`. Patched the
    test to use a minimal transformer stub with `hidden_size` and
    `num_attention_heads`.
  - Modal L40S app `ap-MqL3ZXebN28IDjqArPhAi0` then ran
    `pytest tests/local_tests/tdm/ -v -s` from base commit `7c6ddd8ee` with
    the local patch applied. Result before the later shifted-sigma fix:
    `19 passed in 18.33s`.
  - First two-step real-training smoke app `ap-dFqUVQ8af0L7Ih9ufyiD1D`
    completed training but intentionally failed a post-run assertion:
    generator sigma for raw timestep `750` was `0.750257670879364`. Downloaded
    tracker/config to
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_diag_reviewer_flow_ode_2step_7c6ddd8e_patch_v1/tracker/`.
    The config confirmed `pipeline_config.flow_shift == 8`,
    `pipeline_config.dmd_sample_type == "ode"`, and
    `method.student_sample_type == "ode"`, so the remaining bug was not config
    parsing. The root cause was TDM interpreting explicit YAML timesteps as
    already-shifted scheduler labels when converting to sigmas. Validation
    uses raw timesteps plus shifted sigmas via `scheduler.set_timesteps(...)`.
  - Patched `TDMMethod._timestep_to_sigma(...)` so static flow schedulers
    convert explicit raw timesteps to shifted sigmas directly. Patched the TDM
    SDE rollout path to use the same sigma lookup instead of
    `student.add_noise(...)`, keeping SDE and ODE noising semantics aligned.
    Patched `DmdDenoisingStage.forward(...)` to rebuild the scheduler table
    from `pipeline_config.dmd_denoising_steps` before denoising, since that
    stage replaces `batch.timesteps` with the DMD list.
  - Added regression coverage in
    `tests/local_tests/tdm/test_tdm_method_unit.py` with a shifted fake flow
    scheduler. The test asserts raw explicit steps `[1000, 750, 500, 250]`
    map to shifted sigmas `[1.0, 0.96, 0.888..., 0.727...]` instead of the old
    unshifted `{1.0, 0.75, 0.5, 0.25}` interpretation.
  - Modal L40S app `ap-93a5thhmV4NKH17U918XCC` reran
    `pytest tests/local_tests/tdm/ -v -s` after the shifted-sigma fix. Result:
    `20 passed in 17.36s`.
  - Two-step real-training smoke app `ap-VL9pTwkpEWNVWPTSmpjSpt` ran on
    H100:1 from base commit `7c6ddd8ee` with the local patch and a temporary
    uncommitted diagnostic script. Command used the Wan TDM YAML, staged
    `crush-smol_processed_t2v`, `max_train_steps=2`,
    `method.generator_update_interval=1`, JSONL-only tracking,
    checkpoint saves disabled, and validation disabled via
    `callbacks.validation.every_steps=0`.
  - The successful H100 smoke printed:
    `SMOKE_TIMESTEPS [750.0, 750.0]`,
    `SMOKE_SIGMAS [0.9599999785423279, 0.9599999785423279]`,
    `SMOKE_TRAJECTORY_INDICES [1.0, 1.0]`,
    `SMOKE_FLOW_SHIFT 8`, and `SMOKE_DMD_SAMPLE_TYPE ode`. The only warning
    was the known non-fatal NCCL `destroy_process_group()` warning after
    process exit.
  - Downloaded the successful smoke tracker/config to
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_diag_reviewer_flow_ode_2step_7c6ddd8e_patch_v2/tracker/`.
    Local summary: `8` JSONL rows, `2` generator rows, generator timesteps
    `[750.0, 750.0]`, generator sigmas
    `[0.9599999785423279, 0.9599999785423279]`, trajectory indices
    `[1.0, 1.0]`, and tracker config retained `flow_shift=8`,
    `dmd_sample_type=ode`, `student_sample_type=ode`.
  - The temporary `scripts/diagnostics/issue775_tdm_smoke.py` file used only
    for the Modal smoke was removed before committing and is not part of the
    final branch diff.
- Final local/static gate before commit:
  - `git diff --check`: passed.
  - `python -m py_compile` on all changed Python files: passed.
  - Running changed-file `pre-commit run --files ...` directly in the issue
    worktree applied yapf formatting and then hit the known mypy realpath
    issue `issue-775-tdm is not a valid Python package name`.
  - Recreated the final diff in valid-path clone
    `/tmp/fastvideo_worktrees/issue_775_tdm_clone_final2`; changed-file
    `pre-commit run --files ...` passed all hooks there, including mypy.
  - Pre-commit GitHub refresh: `gh` identity is `macthecadillac`; issue #775
    remains open and assigned to `macthecadillac`; comments are unchanged; no
    open PRs matched targeted search `775 OR TDM`.
- After validation, commit and push with GPG signing, then rerun Stage 3
  review/adjudication because this is a later user-directed code change after
  implementation had already begun. Do not launch the next 500-step DGX
  apples-to-apples run until the fresh committed branch has cleared that loop.
- Signed code commit and push:
  - Commit `ce68de9f2a08126b649db1c99bb212ef5bea39fa`
    `[fix]: align TDM trajectory schedule and objective`.
  - `git log -1 --show-signature` verified a good signature from
    `Mac Lee <macthecadillac@gmail.com>` using subkey
    `9970C3F2BC145193A5C12AAD4C1D75FF3B58866D`.
  - Pushed `issue/775-tdm` to `origin` (`macthecadillac/FastVideo`);
    push succeeded with only the known non-fatal SSH `known_hosts`
    cross-device-link warnings.
- Fresh Stage 3 review loop after this later user-directed code change:
  - Spawned review-code sub-agent `019f459c-4300-7211-b2a2-d6bd5bee4a1b`
    (`Popper`) to review committed branch `macthecadillac/FastVideo`
    `issue/775-tdm` at `ce68de9f2a08126b649db1c99bb212ef5bea39fa`
    for issue #775.
  - Reviewer `Popper` completed with three findings:
    1. Medium: `warp_denoising_step=True` double-applies the flow-shift
       schedule in TDM because warped steps are already scheduler timesteps
       and then `_timestep_to_sigma(...)` applies static flow shift again.
       Recommendation: reject `warp_denoising_step=True` for TDM or derive
       warped sigmas directly from scheduler sigmas; add shifted-scheduler
       regression coverage.
    2. Medium: production-path coverage is still thin for a new training
       method. Recommendation: add lightweight `fastvideo/tests/train/...`
       coverage for config/build/validation propagation.
    3. Low: the branch still commits the temporary handoff. Recommendation:
       remove before PR creation.
  - Per Stage 3, these findings were passed verbatim to a separate
    adjudicator/fixer agent for independent decision and any accepted fixes.
- Independent adjudicator/fixer pass started in the same worktree at
  `/tmp/fastvideo-worktrees/issue-775-tdm`, with `gh` verified as
  `macthecadillac`. Issue #775 remains open with only the maintainer-interest
  and stale-bot comments; targeted open PR searches for `775` and
  `issue/775-tdm` returned `[]`.
- Adjudicator decisions:
  - Accepted reviewer finding 1. The committed `warp_denoising_step=True`
    path converted raw TDM steps through `student.noise_scheduler.timesteps`,
    then `_timestep_to_sigma(...)` treated those shifted labels as raw Wan
    timesteps for static flow-shift schedulers. This would apply flow shift a
    second time. Patched TDM to preserve warped scheduler timestep labels as
    floats and to resolve exact scheduler-table timestep labels directly from
    `scheduler.sigmas` before using the raw-timestep static flow formula.
    Added shifted-scheduler regression coverage for warped steps.
  - Accepted reviewer finding 2 as a coverage gap. Added a package-level
    `fastvideo/tests/train/...` regression that loads the shipped Wan TDM YAML,
    builds `TDMMethod` through `build_from_config(...)` using lightweight Wan
    stubs, verifies inherited scheduler shift and raw-step sigma mapping, and
    exercises validation `sampling_timesteps` propagation into a fake
    `WanDMDPipeline`/DMD inference args.
  - Rejected reviewer finding 3 for the current stage. The handoff is still
    required because no PR exists and Stage 4 has not been explicitly
    requested. It should be removed immediately before PR creation, not during
    this adjudicator/fixer pass.
- Validation after this patch:
  - First Modal L40S targeted pytest app `ap-347yxveZpbUrHjSreDTAut`
    correctly applied the local patch and ran the intended command, but failed
    one new test because the fake validation dataset bypassed
    `ValidationDataset.__iter__` and did not include the real `prompt` alias.
    This was a test-fixture bug, not a product-code failure.
  - Patched the fake validation dataset to include both `caption` and
    `prompt`.
  - Modal L40S app `ap-gDHLgh0a5mrwRaJcdNcfVy` reran
    `pytest tests/local_tests/tdm/ fastvideo/tests/train/methods/test_tdm_config_path.py -q`
    from base commit `ce68de9f2a08126b649db1c99bb212ef5bea39fa` with the
    local code/test patch applied. Result: `23 passed in 2.87s`.
  - Modal L40S app `ap-b6zH5CoNCCcZDhpZ3jwHV4` ran
    `pre-commit run --files fastvideo/train/methods/distribution_matching/tdm.py tests/local_tests/tdm/test_tdm_method_unit.py fastvideo/tests/train/methods/test_tdm_config_path.py .agents/handoffs/issue-775-handoff.md`
    from the same base commit with the local patch applied. Result: passed
    `yapf`, `ruff`, `codespell`, `mypy`, filename-space, and suggestion
    hooks; PyMarkdown/actionlint had no files to check.
  - Final GitHub refresh before commit/push: issue #775 remained open and
    assigned to `macthecadillac`; issue comments were unchanged; targeted open
    PR searches for `775 OR TDM` and head `issue/775-tdm` returned `[]`; no
    PR state was changed.
- Signed code commit and push:
  - Commit `c3613d4dc5d5b6298c2822b1cb6744cb40efae0f`
    `[fix]: avoid TDM warped schedule double shift`.
  - `git log -1 --show-signature` verified a good signature from
    `Mac Lee <macthecadillac@gmail.com>` using subkey
    `9970C3F2BC145193A5C12AAD4C1D75FF3B58866D`.
  - Pushed `issue/775-tdm` to `origin` (`macthecadillac/FastVideo`);
    push succeeded with only the known non-fatal SSH `known_hosts`
    cross-device-link warnings.
- Because this adjudicator accepted findings and changed code, the parent
  Stage 3 loop continued with a fresh `review-code` pass against the updated
  committed branch. No PR was opened, and the handoff remains active.
- Parent verified local worktree clean and synced at
  `869bd908057fa0fbebed4e0b2b25aa269c64b345`; latest signature verified good
  from `Mac Lee <macthecadillac@gmail.com>`.
- Spawned fresh review-code sub-agent
  `019f45b8-67b3-7b81-b22f-de0c3c867b0d` (`Banach`) to review updated branch
  head `869bd908057fa0fbebed4e0b2b25aa269c64b345`.
- Reviewer `Banach` completed with two findings:
  1. High: TDM uses inconsistent timestep-to-sigma mappings during training.
     `_timestep_to_sigma(...)` maps raw YAML steps such as `750` through flow
     shift to sigma about `0.96`, but `_student_trajectory()` passes raw
     timestep `750` directly into `student.predict_x0()`, whose
     `pred_noise_to_pred_video(...)` lookup can resolve sigma around `0.75`
     from the role scheduler's shifted timestep table. Recommendation: use one
     authoritative schedule; either convert configured raw steps to scheduler
     timestep labels before all `predict_x0()` calls, or make x0
     conversion/noising consume the same sigmas as `_timestep_to_sigma()`.
     Add shifted Wan scheduler regression coverage.
  2. Medium: shipped TDM example enables validation every 50 steps without
     `offload_training_state` or `unload_pipeline_after_validation`, even
     though TDM keeps student/teacher/critic resident and validation loads an
     inference pipeline. Recommendation: enable both offload settings in the
     TDM validation block, or disable validation by default and document the
     separate command.
- Per Stage 3, these second-review findings were passed verbatim to another
  separate adjudicator/fixer agent for independent decision and any accepted
  fixes.
- User then raised a new possible quality culprit: effective batch size may
  have been too small and asked to try `bz = 4` or `8`, with gradient
  accumulation acceptable if true larger batch OOMs. Current config and prior
  DGX runs used `training.data.train_batch_size=1` and
  `training.loop.gradient_accumulation_steps=1`, so this is an experiment
  variable not previously covered. The next DGX action is a short detached
  Docker smoke at true `train_batch_size=4`; if it fails with a CUDA OOM
  signature, retry equivalent effective batch 4 with
  `train_batch_size=1` and `gradient_accumulation_steps=4`. Do not run a long
  500-step batch-size comparison until this smoke confirms memory/numerics and
  the still-open Stage 3 scheduler-consistency review finding is resolved or
  explicitly accepted as residual risk.
- Clarified data/model path: `data/Wan-Syn_77x448x832_600k` is the configured
  training dataset path. The Wan model weights are loaded from the
  `models.*.init_from` entries, currently
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, and cached separately under the HF model
  cache. Earlier DGX comparison runs overrode the dataset path to the staged
  `wlsaidhi/crush-smol_processed_t2v` parquet dataset under the run-local
  HF cache; the pretrained model snapshot was not downloaded from
  `data/Wan-Syn_77x448x832_600k`.
- User clarified that the next run should use
  `data/Wan-Syn_77x448x832_600k` instead of the staged
  `wlsaidhi/crush-smol_processed_t2v` dataset used by the earlier DGX/Modal
  diagnostics, and said to restart the job if necessary. DGX inspection showed
  no active issue-775 containers and no existing Wan-Syn dataset copy under
  the checked common paths.
- Launched replacement detached DGX Spark Docker job:
  - Container: `issue775_tdm_wansyn_bsz4_smoke_20260709100811`, id
    `dd33083db3388aa1ed6bd7297617f0729c3b000819bd5950544152c17bf9e0f0`.
  - Run root:
    `/home/mac/fastvideo-runs/issue-775/tdm_wansyn_bsz4_smoke_486e4ff_20260709100811`.
  - Log:
    `/home/mac/fastvideo-runs/issue-775/tdm_wansyn_bsz4_smoke_486e4ff_20260709100811/logs/wansyn_batch_size_smoke.log`.
  - Shared dataset target:
    `/home/mac/fastvideo-runs/issue-775/datasets/Wan-Syn_77x448x832_600k`.
  - The run symlinks that shared dataset into
    `/workspace/run/FastVideo/data/Wan-Syn_77x448x832_600k`, then launches
    training with explicit `--training.data.data_path
    data/Wan-Syn_77x448x832_600k`.
  - The process tree was verified with `docker top` as user `mac`; the
    container is detached, uses `--user 1006:1006`, and uses
    `fastvideo-dev-nonroot:issue775`.
  - First observed status:
    `docker inspect` reported `status=running running=true exit=0 oom=false`.
    The job had cloned commit
    `486e4ffd71d2436f74436b5a685a442b54d0fa7f`, entered dataset staging, and
    started `snapshot_download` for
    `FastVideo/Wan-Syn_77x448x832_600k`; the dataset directory had reached
    about `2.1G`.
  - Follow-up status at about `2026-07-09T10:12Z`: still running with
    `oom=false`, still in dataset staging. The log showed
    `Fetching ... files` through file `114`, and the shared dataset directory
    had reached about `17G`.
  - Final status checked on `2026-07-10`: container exited successfully with
    `status=exited running=false exit=0 oom=false`, started
    `2026-07-09T10:09:49Z`, finished `2026-07-09T15:45:06Z`.
  - Wan-Syn dataset staging completed. The dataset path now reports about
    `1.6T`, with `13877` parquet files and `111016` rows per SP group.
  - True batch 4 passed; no gradient-accumulation fallback was used.
    Metrics path:
    `/home/mac/fastvideo-runs/issue-775/tdm_wansyn_bsz4_smoke_486e4ff_20260709100811/output_true_bsz4/tracker/metrics.jsonl`.
    Metrics summary: `8` JSONL rows, steps `1..2`, `2` loss rows,
    `0` nonfinite scalar metrics.
  - Step 1: `step_time_sec=488.5659004969`,
    `total_loss=1.2896826267`, `generator_loss=0.8548319340`,
    `fake_score_loss=0.4348507524`,
    `grad_norm/student=1.4613537788`, and
    `grad_norm/critic=0.2916031182`.
  - Step 2: `step_time_sec=446.0986256851`,
    `total_loss=1.0857334137`, `generator_loss=0.8703663349`,
    `fake_score_loss=0.2153671086`,
    `grad_norm/student=0.5097031593`, and
    `grad_norm/critic=0.1462094337`.
  - The actual training command confirmed the requested dataset path:
    `training.data.data_path='data/Wan-Syn_77x448x832_600k'`,
    `training.data.train_batch_size=4`, and
    `training.loop.gradient_accumulation_steps=1`.
  - Planned command after dataset staging: true
    `training.data.train_batch_size=4`,
    `training.loop.gradient_accumulation_steps=1`,
    `training.loop.max_train_steps=2`, JSONL tracker only, no checkpoint
    saves, validation disabled. If true batch 4 fails with a CUDA OOM
    signature, the script retries effective batch 4 with
    `train_batch_size=1` and `gradient_accumulation_steps=4`.
- DGX Spark true-batch-size smoke completed:
  - Detached non-root Docker container:
    `issue775_tdm_bsz4_smoke_20260709074338`, id
    `f7e38c1d1db9c1e0b19f8299e2fc472275829b3d0d4b98eb2b4fec54d6d72057`.
    `docker inspect` after completion reported
    `status=exited running=false exit=0 oom=false`, finished
    `2026-07-09T08:09:31Z`.
  - Run root:
    `/home/mac/fastvideo-runs/issue-775/tdm_bsz4_smoke_869bd90_20260709074338`.
    Main log:
    `/home/mac/fastvideo-runs/issue-775/tdm_bsz4_smoke_869bd90_20260709074338/logs/batch_size_smoke.log`.
    Metrics:
    `/home/mac/fastvideo-runs/issue-775/tdm_bsz4_smoke_869bd90_20260709074338/output_true_bsz4/tracker/metrics.jsonl`.
  - The process tree was verified with `docker top` as user `mac` before and
    during training. The run used detached Docker with `--user 1006:1006` and
    the DGX-local `fastvideo-dev-nonroot:issue775` derivative of the required
    FastVideo Docker image.
  - The command checked out committed branch head
    `869bd908057fa0fbebed4e0b2b25aa269c64b345`, set
    `training.data.train_batch_size=4`,
    `training.loop.gradient_accumulation_steps=1`,
    `training.loop.max_train_steps=2`, `method.generator_update_interval=1`,
    JSONL-only tracking, no checkpoint saves, and validation disabled.
  - The dataset was the same staged `wlsaidhi/crush-smol_processed_t2v`
    parquet dataset used by the previous DGX comparison run. The log confirms
    the full Wan model snapshot was downloaded from
    `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` into the run-local HF model cache,
    not from `data/Wan-Syn_77x448x832_600k`; the HF cache reached about `28G`.
  - True batch 4 fit memory and did not trigger the scripted gradient
    accumulation fallback. Metrics summary: `8` JSONL rows, steps `1..2`,
    `2` loss rows, `0` nonfinite scalar metrics.
  - Step 1 took `518.659s` with
    `total_loss=1.4091826677`, `generator_loss=0.9760555029`,
    `fake_score_loss=0.4331271946`,
    `grad_norm/student=0.5851445198`, and
    `grad_norm/critic=0.3069764972`.
  - Step 2 took `441.647s` with
    `total_loss=1.2040213346`, `generator_loss=0.9869136810`,
    `fake_score_loss=0.2171076387`,
    `grad_norm/student=0.4463202953`, and
    `grad_norm/critic=0.1409958750`.
  - Practical implication: a 500 optimizer-step true-batch-4 run would be
    roughly `61` hours from the step-2 steady-state time, or `67` hours from
    the two-step mean, before validation/visual generation. A sample-count
    matched comparison to the prior 500-step batch-1 run would be closer to
    `125` optimizer steps at true batch 4, but that is not optimizer-update
    matched. Do not launch a long batch-size run without explicitly choosing
    which comparison basis to use.

## 2026-07-06 Resume Update

### 2026-07-06 06:54 UTC Objective / Schedule Investigation

- Resumed after interruption from recreated worktree
  `/tmp/fastvideo-worktrees/issue-775-tdm` on branch `issue/775-tdm`.
  Worktree was clean at pushed commit `c435ddf01`.
- User selected the next Stage 2 direction: investigate TDM objective,
  sigma schedule, and weighting before spending on a longer interval-1
  convergence run.
- User added an execution constraint for future long training runs: run them
  on DGX Spark, and make sure a connection interruption cannot kill the job.
  Any DGX Spark training must therefore use the required Docker image
  `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest`
  and be launched detached/persistent, for example a named detached Docker
  container and durable log/output paths, optionally supervised from tmux.
  Do not run a long DGX job directly attached to the SSH session.
- Refreshed GitHub state with `gh` as `macthecadillac`: issue #775 remains
  open, assigned to `macthecadillac`, with the same two comments already
  recorded below. Narrow open PR searches for `775` and `TDM` both returned
  `[]`; no PR draft status was changed.
- Code inspection findings for the current objective/schedule:
  - `TDMMethod._student_trajectory` always routes generator gradients through
    `trajectory.final_clean`, i.e. only the final prediction in the 4-step
    rollout carries gradients.
  - `TDMMethod._tdm_generator_loss` then samples a uniform training timestep
    from the full scheduler range via `_sample_training_timestep`, independent
    of the 4-step TDM rollout schedule. This means the student is rolled out
    on `[1000, 750, 500, 250]` but the generator score target is usually
    evaluated at off-schedule timesteps.
  - `noise_interval_mode: separate` excludes terminal `sigma=1.0` fake-score
    targets. For the current 4-step schedule, separate mode only produces
    target sigmas among `0.5` and `0.75`; terminal targets are always absent.
  - The completed full interval-1 tracker showed
    `tdm/fake_score/snr_weight == 1.0` for every loss row. With target sigmas
    `0.5` and `0.75`, the flow SNR is at most `1.0`, so the configured
    `snr_clip: 5.0` is inert under the current schedule.
  - The existing fake-score metrics expose sigma pair and weighting behavior,
    but generator loss currently does not log its sampled timestep/sigma,
    delta magnitude, or normalization denominator. That makes it hard to
    distinguish bad convergence from an off-schedule or poorly scaled
    generator objective.
- Immediate implementation direction: keep changes narrow and directly used.
  Add generator-objective diagnostics and align the TDM generator score
  timestep with the active TDM denoising schedule instead of sampling uniformly
  over the full training grid. This directly tests the strongest schedule
  mismatch without adding an unused config surface.
- Patch applied in this work segment:
  - `fastvideo/train/methods/distribution_matching/tdm.py` now samples the
    generator score timestep from `_get_denoising_step_list(...)` instead of
    the full shifted training grid.
  - `TDMMethod._tdm_generator_loss(...)` now returns focused generator
    diagnostics on student-update steps:
    `tdm/generator/timestep`, `tdm/generator/sigma`,
    `tdm/generator/raw_delta_abs_mean`,
    `tdm/generator/target_delta_abs_mean`,
    `tdm/generator/normalization_denom`, and
    `tdm/generator/normalize_delta`.
  - `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`
    documents that generator score timesteps use `tdm_denoising_steps`.
  - `tests/local_tests/tdm/test_tdm_method_unit.py` now asserts generator
    metrics are emitted on update steps, omitted on skipped generator-update
    steps, and that `_sample_training_timestep(...)` only returns configured
    TDM steps.
- Local non-test checks run:
  - `git diff --check`: passed.
  - `python -m py_compile fastvideo/train/methods/distribution_matching/tdm.py tests/local_tests/tdm/test_tdm_method_unit.py`:
    passed.
  - `awk 'length($0) > 120 ...'` on changed code/config/test files:
    no overlong lines reported.
- Remote validation still needed: Modal L40S `pytest tests/local_tests/tdm/ -v -s`
  through the `interleavethinker` launcher, then a short Wan TDM smoke or
  diagnostic run with JSONL tracking to confirm generator metric rows and
  schedule sampling in real training.
- Remote unit validation completed:
  - Modal app: `ap-jmEzm2lknkXnacVQgNYTmZ`.
  - Launcher: `/tmp/fastvideo-worktrees/interleavethinker-modal/fastvideo/tests/modal/launch_l40s_job.py`.
  - Base commit: `c435ddf01e4e408ce843434aeee3b01f2d6e81cc` from
    `https://github.com/macthecadillac/FastVideo.git`.
  - Local patch applied for `tdm.py`, `test_tdm_method_unit.py`, and
    `tdm_t2v_lora.yaml`.
  - Command: `pytest tests/local_tests/tdm/ -v -s`.
  - Result: `11 passed, 14 warnings in 19.89s`.
  - New coverage includes schedule-aligned generator timestep sampling and
    generator diagnostics on student-update steps.
- Remote real-training smoke completed:
  - Modal app: `ap-M7bm8Y7TP7sQ3K7j4M1ZkC`.
  - Base commit: `c435ddf01e4e408ce843434aeee3b01f2d6e81cc` plus the same
    local patch as the unit validation.
  - Command: 4x L40S `torchrun --standalone --nproc_per_node=4 -m
    fastvideo.train.entrypoint.train` with the Wan TDM config,
    staged `crush-smol_processed_t2v` dataset, `max_train_steps=2`,
    `method.generator_update_interval=1`, JSONL-only tracking, checkpoint
    saves disabled, validation disabled, and output root
    `/root/data/tdm_diag_schedule_metrics_c435ddf_patch`.
  - Result: completed successfully, committed the Modal volume, and produced
    JSONL metrics. Non-fatal NCCL `destroy_process_group()` warnings matched
    prior Modal training runs.
  - Downloaded metrics to
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_diag_schedule_metrics_c435ddf_patch/metrics.jsonl`.
  - Metrics summary: `8` JSONL rows; `2` generator/loss rows; nonfinite
    scalar metrics `0`. Generator timestep/sigma pairs were
    `(750.0, 0.75)` at step 1 and `(500.0, 0.5)` at step 2, confirming real
    training now samples generator score timesteps from the configured TDM
    grid. The same rows included the new generator diagnostics and retained
    fake-score metrics. Fake-score target sigmas were `0.75` and `0.5`;
    `tdm/fake_score/snr_weight` remained `1.0`, consistent with the earlier
    finding that `snr_clip: 5.0` is inert for these target sigmas.
- Changed-file lint/type checks:
  - Initial sandboxed `uvx pre-commit run --files ...` failed because uv
    could not write to `/home/toolbox/.cache/uv` inside the sandbox.
  - Escalated rerun of
    `uvx pre-commit run --files fastvideo/train/methods/distribution_matching/tdm.py tests/local_tests/tdm/test_tdm_method_unit.py examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml .agents/handoffs/issue-775-handoff.md`:
    `yapf`, `ruff`, `codespell`, filename-space check, and suggestion hooks
    passed; PyMarkdown/actionlint had no files to check; mypy hook failed
    before checking files with the known worktree-name issue
    `issue-775-tdm is not a valid Python package name`.
  - Direct mypy fallback:
    `uvx mypy --explicit-package-bases fastvideo/train/methods/distribution_matching/tdm.py`
    passed with `Success: no issues found in 1 source file`.
  - Follow-up `git diff --check`: passed.
- Signed code commit and push:
  - Commit: `9f9c52d7cc1617a266209d85712f5576d117a045`
    `[fix]: align TDM generator score schedule`.
  - `git log -1 --show-signature` verified a good signature from
    `Mac Lee <macthecadillac@gmail.com>` using subkey
    `9970C3F2BC145193A5C12AAD4C1D75FF3B58866D`.
  - Pre-push GitHub refresh: `gh` identity `macthecadillac`; issue #775
    unchanged/open; narrow open PR searches for `775` and `TDM` returned
    `[]`.
  - Pushed to `origin/issue/775-tdm`. Push succeeded with non-fatal
    `known_hosts` cross-device-link warnings from the local SSH hostfile
    update path.
- Next validation direction after this commit: inspect DGX Spark availability
  and launch any longer schedule-aligned interval-1 convergence run only as a
  detached Docker job with durable logs/output, never as a foreground SSH
  process.
- DGX Spark inspection after push:
  - Connected with SSH using task-local known-hosts file
    `/tmp/dgx_spark_known_hosts` after the shared SSH known-hosts file rejected
    the host key.
  - Hostname: `spark-1a51`.
  - Docker is installed: `Docker version 29.2.1, build a5c7197`.
  - GPU query reports `NVIDIA GB10`, driver `580.159.04`; memory field is
    `N/A`, consistent with GB10 unified-memory reporting.
  - `tmux 3.4` is installed.
  - Disk at `/home/mac`: `3.7T` total, `2.9T` available.
  - Current blocker: user `mac` cannot access Docker directly:
    `permission denied while trying to connect to the docker API at
    unix:///var/run/docker.sock`.
  - `id` shows `mac` is in groups `mac` and `sudo`, but not `docker`.
    `/var/run/docker.sock` is owned by `root:docker` with mode `srw-rw----`.
  - `sudo -n docker --version` fails with `sudo: a password is required`.
  - Therefore the longer DGX Spark run was **not launched**. To proceed while
    honoring the project rule that DGX workloads run in Docker, the `mac`
    user needs Docker-socket access, passwordless Docker sudo, or another
    approved way to start the required image
    `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest`.
  - When Docker access is fixed, launch the training as a detached named
    container (or from a detached tmux session that starts a named detached
    container) with logs under a durable directory such as
    `/home/mac/fastvideo-runs/issue-775/`. Do not run long training as a
    foreground SSH-attached process.
- DGX Spark retry after user added `mac` to the `docker` group:
  - Rechecked GitHub state with `gh` as `macthecadillac`: issue #775 remains
    open and assigned to `macthecadillac`; targeted open PR searches for
    `775` and `TDM` returned `[]`.
  - Reconnected to DGX Spark with the task-local known-hosts file
    `/tmp/dgx_spark_known_hosts`. `id` now reports group membership
    `mac`, `sudo`, and `docker`, and `docker ps` works without sudo.
  - Verified the required project image is present on the host:
    `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest`,
    digest
    `sha256:f57024b64eb582f4b5c2b78ebf8f53e747603bb4ab79eda1c18c583eefe3b280`.
  - Verified the image with `--gpus all`; inside the container
    `/opt/venv/bin/python` reports Python `3.12.13`, Torch
    `2.12.0+cu130`, and `torch.cuda.is_available() == True`.
  - A first detached container,
    `issue775_tdm_schedule_500_96af270`, was launched under
    `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270` but stalled
    during a full git clone. It was stopped with `docker stop`; its run
    directory and stopped container were left intact for audit.
  - Current active run:
    container `issue775_tdm_schedule_500_96af270_v2`, id
    `d328cc9ceb7be4da83da6096f38104c61d33d9baaac847e9a6eed8875a626db1`.
    It is launched with `docker run -d`, so SSH disconnection will not kill
    the job. Host-mounted durable state lives under
    `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2`.
  - The current container uses commit
    `96af270abb97b9c5908f8d867ee3601205b758e4`, branch `issue/775-tdm`, and
    the required Docker image. It shallow-clones
    `https://github.com/macthecadillac/FastVideo.git`, installs editable
    with `python -m pip install --no-deps -e .`, downloads
    `wlsaidhi/crush-smol_processed_t2v`, and runs a 1-GPU Wan TDM interval-1
    command to `max_train_steps=500` with JSONL tracking, checkpoints every
    `100` steps, and validation every `100` steps.
  - Logs:
    `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/logs/training.log`.
    Tracker:
    `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/output/tracker/`.
  - Startup validation completed. The first two training steps completed and
    wrote finite JSONL rows. Step 1 had `grad_norm/student=0.0022551888`,
    `grad_norm/critic=0.0015945135`, `fake_score_loss=0.0012508066`,
    `generator_loss=0.2636678517`, and `total_loss=0.2649186552`.
    The new schedule diagnostics appeared in real training at step 1:
    `tdm/generator/timestep=500.0`, `tdm/generator/sigma=0.5`,
    `tdm/generator/normalization_denom=0.033447265625`,
    `tdm/generator/raw_delta_abs_mean=0.0187694486`, and
    `tdm/generator/target_delta_abs_mean=0.5606355071`.
    Step 2 also wrote finite rows with `grad_norm/student=0.0288033448`,
    `grad_norm/critic=0.0017635319`, `fake_score_loss=0.0005794067`,
    `generator_loss=0.6215059757`, and `total_loss=0.6220853925`.
    Its generator diagnostics sampled another configured TDM step:
    `tdm/generator/timestep=1000.0`, `tdm/generator/sigma=1.0`,
    `tdm/generator/normalization_denom=0.9140625`,
    `tdm/generator/raw_delta_abs_mean=0.8226878047`, and
    `tdm/generator/target_delta_abs_mean=0.9000012875`.
  - Last observed DGX status before this handoff update:
    `docker inspect` reports `running true 0`; the progress bar had reached
    `Steps: 2/500`. Leave the detached container running unless the user asks
    to stop or relaunch it.
- DGX Spark status check on 2026-07-07:
  - Rechecked GitHub state before recording/pushing this update: `gh` identity
    is `macthecadillac`; issue #775 remains open and assigned to
    `macthecadillac`; targeted open PR searches for `775` and `TDM` returned
    `[]`.
  - Container `issue775_tdm_schedule_500_96af270_v2` is no longer running.
    `docker inspect` reports
    `status=exited running=false exit=1 oom=false error=`, started
    `2026-07-06T07:45:26Z`, finished `2026-07-06T10:57:15Z`.
  - Docker events in the termination window show only the container `die`
    event with `exitCode=1`; no Docker OOM event or explicit kill event was
    reported by Docker.
  - The run reached step `100/500`, saved
    `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/output/checkpoint-100`,
    started step-100 validation, offloaded optimizer, teacher, and critic
    state, then received `SIGTERM` while loading the validation text encoder.
    Log root-cause line:
    `Signal 15 (SIGTERM) received by PID 352`.
  - Tracker summary:
    `400` JSONL rows, steps `1..100`, `100` loss rows, `0` nonfinite scalar
    metrics. Mean `step_time_sec` was `105.139775953`; last-10-step mean was
    `104.977194918`. Last loss row at step `100`:
    `total_loss=0.3323473930`, `generator_loss=0.3318239748`,
    `fake_score_loss=0.0005234162`.
  - Durable artifacts currently present in the run output include
    `checkpoint-100`, `tracker/metrics.jsonl`, `tracker/artifacts.jsonl`,
    `tracker/config.json`, and step-0 validation videos. There are no
    step-100 validation videos because the process was terminated during
    validation setup.
  - Suggested next action: resume from `checkpoint-100` in a new detached
    Docker container, but avoid repeating the same validation failure mode.
    The narrowest retry is to resume to step 200 with validation disabled or
    moved later, then separately run validation from a checkpoint if needed.
- DGX Spark hardened non-root resume on 2026-07-07:
  - User requested the resume be hardened so the host process appears as
    `mac` instead of `root` in process listings.
  - Built a DGX-local derivative image tagged
    `fastvideo-dev-nonroot:issue775` from the required base image
    `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest`.
    Image id after rebuild: `e97970511b99`. The derivative only:
    (1) copies the uv-managed Python runtime out from under `/root` to
    `/opt/uv-python/cpython`, retargets `/opt/venv/bin/python*`, and
    (2) adds `mac` as UID/GID `1006` in `/etc/passwd` and `/etc/group`.
    This is still a Docker workload and does not use host `sudo`.
  - Verified the image under `--user 1006:1006 --gpus all`: `id` resolves
    `uid=1006(mac) gid=1006(mac)`, `getpass.getuser()` returns `mac`,
    `/opt/venv/bin/python` runs Python `3.12.13`, Torch is
    `2.12.0+cu130`, and CUDA is available.
  - Repaired ownership of the existing host-mounted run directory with a
    short Docker-launched `chown -R 1006:1006 /workspace/run`; no host `sudo`
    was used. Verified key paths now report `mac:mac`, including
    `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/output`,
    `output/checkpoint-100`, and `cache/hf`.
  - Installed the resume script at
    `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/run_resume_nonroot.sh`.
    It sets writable `HOME`, HF, XDG, Torch, matplotlib, and temp caches under
    `/workspace/run`; runs from source via
    `PYTHONPATH=/workspace/run/FastVideo`; avoids `pip install -e .`; resumes
    from `/workspace/run/output/checkpoint-100`; keeps checkpoints every
    `100` steps; resumes to `max_train_steps=500`; and disables validation
    with `callbacks.validation.every_steps=0` to avoid repeating the
    step-100 validation SIGTERM path.
  - First non-root launch,
    `issue775_tdm_schedule_500_96af270_v3_nonroot`, exited before training
    because the first derivative image had UID `1006` but no `/etc/passwd`
    entry, causing PyTorch/TorchInductor username lookup to fail:
    `KeyError: 'getpwuid(): uid not found: 1006'`. The image was rebuilt with
    the `mac` user entry before retrying.
  - Active resumed container:
    `issue775_tdm_schedule_500_96af270_v4_nonroot`, id
    `df15cb858a67773b7eaadafdcac7544f0dc4e47ba77235fcb8954748be05f824`,
    launched detached with `--user 1006:1006`, `--gpus all`, the existing run
    directory bind-mounted to `/workspace/run`, and image
    `fastvideo-dev-nonroot:issue775`.
  - Host process ownership was verified with `docker top`: the wrapper
    `bash`, `tee`, `torchrun`/`pt_elastic`, worker `python3`, and Python
    resource tracker all appear as `mac`, UID `1006`, GID `1006`.
  - Resume log:
    `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/logs/training_resume_nonroot.log`.
    The log confirms `Checkpoint loaded; resuming from step=100`, RNG restored
    from `checkpoint-100`, and validation is instantiated with
    `every_steps: 0`.
  - Step `101` completed with finite JSONL rows:
    `grad_norm/student=0.0056205532`,
    `grad_norm/critic=0.0017859151`,
    `fake_score_loss=0.0010663703`,
    `generator_loss=0.3963286579`,
    `total_loss=0.3973950148`,
    `tdm/generator/timestep=500.0`, and `tdm/generator/sigma=0.5`.
  - Last observed status before this handoff update:
    `docker inspect` reports
    `status=running running=true exit=0 oom=false`; the progress bar had
    reached the resumed step `101` region. Leave this detached v4 container
    running unless the user asks to stop it.
  - Completion check on 2026-07-07: the same container now reports
    `status=exited running=false exit=0 oom=false`, started
    `2026-07-07T02:33:46.152754329Z`, finished
    `2026-07-07T14:17:37.144901209Z`.
  - Final log tail covered steps `482..500`, saved
    `/workspace/run/output/checkpoint-500`, emitted the known duplicate final
    `checkpoint-500` overwrite warning from final teardown, and ended with
    `Training completed`.
  - Completed-run tracker summary from
    `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/output/tracker/metrics.jsonl`:
    `2000` JSONL rows, steps `1..500`, `500` loss rows, and `0` nonfinite
    scalar metrics. Mean `step_time_sec` was `105.0814856065`; last-10-step
    mean was `105.1063260978`. Last loss row at step `500`:
    `total_loss=0.6324374079704285`,
    `generator_loss=0.6307809352874756`,
    `fake_score_loss=0.0016564970137551427`,
    `tdm/generator/timestep=1000.0`, and `tdm/generator/sigma=1.0`.
  - Durable checkpoints now present in the host-mounted output:
    `checkpoint-100`, `checkpoint-200`, `checkpoint-300`, `checkpoint-400`,
    and `checkpoint-500`, each with `metadata.json`.
  - No new validation videos beyond the earlier step-0 artifacts are expected
    from this non-root resume because `callbacks.validation.every_steps=0`
    was intentionally set to avoid repeating the step-100 validation SIGTERM
    failure path.
  - Refreshed GitHub state before recording this completion: `gh` identity is
    `macthecadillac`; issue #775 remains open and assigned to
    `macthecadillac`; targeted open PR searches for `775` and `TDM` both
    returned `[]`; no PR draft status was changed.
  - Checkpoint-500 video validation completed on DGX Spark after the user
    asked for a validation run:
    - Student validation used the FastVideo training validation callback from
      `/workspace/run/output/checkpoint-500`, generated four TDM 4-step
      videos with `sampling_steps=[4]`,
      `sampling_timesteps=[1000,750,500,250]`, and
      `guidance_scale=6.0`, then exited nonzero only because a following
      inline teacher-generation script tried strict JSON parsing on the
      repo's trailing-comma `validation_4.json`.
    - Student validation container:
      `issue775_tdm_step500_validation_20260707_nonroot`, id
      `9a4034db1db396d84cac369ae24cf684f0dbae8c56708cccb932a50e035ca23b`,
      `status=exited running=false exit=1 oom=false`, started
      `2026-07-07T14:53:38.721304274Z`, finished
      `2026-07-07T15:01:47.590903055Z`.
    - Student outputs are saved on DGX at
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/validation_step500/student_tdm_4step/validation_step_500_inference_steps_4_video_{0..3}.mp4`.
      File sizes are approximately `289K`, `1.2M`, `157K`, and `1.3M`.
    - Teacher/base Wan validation used a real script copied to DGX at
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/validation_step500/run_teacher_validation.py`
      because `VideoGenerator` multiprocessing spawn cannot run safely from
      stdin. The script loads the same four prompts through
      `datasets.load_dataset("json", field="data")`, then generates from
      `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` with `height=448`, `width=832`,
      `num_frames=77`, `fps=16`, `num_inference_steps=50`,
      `guidance_scale=6.0`, and seed `1000`.
    - Teacher validation container:
      `issue775_tdm_step500_teacher_validation_v2_20260707_nonroot`, id
      `a5028f5c68ee1c4b4598f6c637c374d08054f7f7bd855770f1212431f879b6df`,
      `status=exited running=false exit=0 oom=false`, started
      `2026-07-07T15:05:11.982770983Z`, finished
      `2026-07-07T15:46:14.116088764Z`.
    - Teacher outputs are saved on DGX at
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/validation_step500/teacher_wan_50step/teacher_wan_50step_prompt{0..3}.mp4`.
      Prompt text/index mapping is saved at
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/validation_step500/teacher_wan_50step/prompts.json`.
      File sizes are approximately `317K`, `832K`, `242K`, and `1.1M`;
      `prompts.json` is approximately `3.0K`.
    - Logs are saved at
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/logs/validation_step500.log`
      for the student callback run and
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_96af270_v2/logs/validation_step500_teacher_v2.log`
      for the successful teacher run.
    - `ffprobe` sanity check on all eight MP4s passed. Every student and
      teacher video reports `832x448`, `77` frames, `16` fps, and duration
      `4.812500s`.
    - Final comparison set includes both student checkpoint-500 4-step videos
      and teacher/base Wan 50-step videos. Technically, the built-in training
      validation callback produced the student videos only; the teacher videos
      were generated separately for comparison using the same prompt set.
  - User visually inspected the checkpoint-500 validation outputs and reported
    all four student clips are still heavily degraded, to the point the frame
    contents are not recognizable. Treat this as a failed quality validation
    signal, not as sufficient evidence to spend on a longer run with the same
    settings.
  - Refreshed GitHub state before the next code change on 2026-07-08:
    `gh` identity is `macthecadillac`; issue #775 remains open and assigned to
    `macthecadillac`; issue comments are still only the maintainer interest
    comment from `zhisbug` and the stale-bot comment; targeted open PR search
    for `775 OR TDM` returned `[]`; no PR draft status was changed.
  - Next hypothesis selected by the user: align generator score timesteps with
    the fake-score target support, not the full TDM rollout grid. Before this
    patch, generator score sampling used every configured TDM denoising step
    (`1000`, `750`, `500`, `250`). Under the shipped `noise_interval_mode:
    separate` schedule, fake-score targets are only the valid non-terminal
    target sigmas with a lower-sigma source, which for the current grid are
    `750` and `500`. Sampling generator guidance at `1000` and `250` therefore
    asks the critic for guidance where it is not trained by the separate-mode
    fake-score objective.
  - Patch applied in this work segment:
    - `fastvideo/train/methods/distribution_matching/tdm.py` now derives a
      generator score timestep list from valid fake-score target support.
      With `noise_interval_mode: separate`, it excludes terminal max-sigma
      and lowest-sigma endpoints; with `noise_interval_mode: to_terminal`, it
      uses the terminal max-sigma timestep.
    - `tests/local_tests/tdm/test_tdm_method_unit.py` now expects separate-mode
      generator score samples only from `{750, 500}` and covers terminal-mode
      sampling.
    - `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`
      documents the derived fake-score-target generator support.
  - Local non-test checks after the patch:
    `git diff --check` passed;
    `python -m py_compile fastvideo/train/methods/distribution_matching/tdm.py tests/local_tests/tdm/test_tdm_method_unit.py`
    passed; changed-file line-length scan reported no lines over 120 chars.
  - Remote validation still needed after this handoff update: Modal L40S
    targeted `pytest tests/local_tests/tdm/ -v -s`, then a short real Wan TDM
    diagnostic run to confirm JSONL generator metrics only contain
    `tdm/generator/timestep` values `750.0` and `500.0` under the shipped
    separate-mode config. Do not launch another long DGX convergence run until
    a short canary shows at least recognizable prompt structure.
  - Remote validation completed after the patch:
    - Modal L40S targeted unit/config/scheduler validation app
      `ap-dkepDHPikkLfo8jyen37kB` ran
      `pytest tests/local_tests/tdm/ -v -s` from base commit
      `ebd07ad8f90f4528c5cc6ceb305c656de2c8d8c2` with the local patch
      applied. Result: `12 passed in 20.86s`.
    - A first 4x L40S real-training smoke app
      `ap-wGA8Ytajt1vBZerX7wUqa0` was queued waiting for `GPU_L40S`
      capacity and was stopped before running any container.
    - Modal H100:1 real-training smoke app
      `ap-v4R2pPiPPxl4zIC6FCoMnP` ran two Wan TDM training steps from the
      same base commit plus local patch. Because this was a smoke after L40S
      capacity was unavailable, it overrode distributed settings to
      `num_gpus=1`, `sp_size=1`, `tp_size=1`, `hsdp_replicate_dim=1`, and
      `hsdp_shard_dim=1`; used the staged
      `crush-smol_processed_t2v` parquet dataset; set
      `method.generator_update_interval=1`; disabled checkpoint saves and
      validation; and used JSONL-only tracking under
      `/root/data/tdm_diag_generator_target_support_20260708_h100_patch`.
    - The H100 smoke completed successfully and committed the Modal volume.
      The final parser assertion printed
      `GENERATOR_TIMESTEPS [500.0, 500.0]` and
      `GENERATOR_SIGMAS [0.5, 0.5]`, satisfying the assertion that generator
      score support is a subset of `{750.0, 500.0}` / `{0.75, 0.5}` and no
      longer includes `1000` or `250`.
    - Downloaded metrics to
      `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_diag_generator_target_support_20260708_h100_patch/metrics.jsonl`.
      Local summary: `8` JSONL rows, `2` loss/generator rows, steps `[1, 2]`,
      generator timesteps `[500.0, 500.0]`, generator sigmas `[0.5, 0.5]`,
      and `0` nonfinite scalar metrics.
    - The only warning was the known non-fatal NCCL
      `destroy_process_group()` shutdown warning after process exit.
  - Changed-file lint/type checks after validation:
    - `uvx pre-commit run --files fastvideo/train/methods/distribution_matching/tdm.py tests/local_tests/tdm/test_tdm_method_unit.py examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml .agents/handoffs/issue-775-handoff.md`
      passed `yapf`, `ruff`, `codespell`, filename-space, and suggestion
      hooks; PyMarkdown/actionlint had no files to check; the mypy hook failed
      before checking files with the known hyphenated worktree basename error
      `issue-775-tdm is not a valid Python package name`.
    - Direct fallback
      `uvx mypy --explicit-package-bases fastvideo/train/methods/distribution_matching/tdm.py`
      passed with `Success: no issues found in 1 source file`.
  - Refreshed GitHub state again before committing/pushing this patch:
    `gh` identity is `macthecadillac`; issue #775 remains open and assigned
    to `macthecadillac`; issue comments are unchanged; targeted open PR search
    for `775 OR TDM` returned `[]`; no PR draft status was changed.
  - Signed code commit and push:
    - Commit: `d21bc5c06ca373695a8e96fb5b16330ef9a67f1b`
      `[fix]: align TDM generator score targets`.
    - `git log -1 --show-signature` verified a good signature from
      `Mac Lee <macthecadillac@gmail.com>` using subkey
      `9970C3F2BC145193A5C12AAD4C1D75FF3B58866D`.
    - Pushed to `origin/issue/775-tdm`. Push succeeded with the known
      non-fatal local `known_hosts` cross-device-link warning.
  - Stage 3 review loop started after the pushed code commit:
    - Spawned fresh read-only `review-code` sub-agent `Boyle`
      (`019f3fde-d6b0-7d41-bebc-5a998da88d42`) to review
      `macthecadillac/FastVideo issue/775-tdm` for issue #775.
    - Reviewer returned three actionable findings:
      (1) `noise_interval_mode="to_terminal"` targets sigma `1.0`, where
      flow SNR is zero and fake-score weights become zero, so terminal-mode
      critic training contributes no gradient while generator guidance samples
      the same untrained terminal target;
      (2) `tests/local_tests/tdm/README.md` contains stale branch/run-specific
      validation history and should be reduced to durable test guidance;
      (3) `fastvideo/train/callbacks/grad_clip.py` still contains
      `debug_log` / `debug_log_steps` scaffolding from earlier diagnostics that
      is not used by the shipped TDM config or tests.
    - Spawned fresh independent adjudicator/fixer sub-agent `Lagrange`
      (`019f3fe9-4d3b-72d2-8bab-9730e2e8e40d`) with only the issue/repo,
      committed branch/commit, and reviewer findings.
    - Adjudicator accepted all three findings and rejected none.
    - Adjudicator implemented and pushed signed commit
      `8d46c5a7c` `[fix]: keep TDM terminal mode trainable`.
      `git log -1 --show-signature` verified a good signature from
      `Mac Lee <macthecadillac@gmail.com>` using subkey
      `9970C3F2BC145193A5C12AAD4C1D75FF3B58866D`.
    - Adjudicator changes:
      - `fastvideo/train/methods/distribution_matching/tdm.py`: terminal mode
        now targets the highest non-terminal sigma rather than exact
        `sigma=1.0`, keeping fake-score critic weights trainable.
      - `tests/local_tests/tdm/test_tdm_method_unit.py`: terminal-mode
        expectations updated and nonzero critic-gradient regression coverage
        added.
      - `tests/local_tests/tdm/README.md`: stale branch/commit/run-specific
        validation history removed; durable test scope and Modal command
        guidance retained.
      - `fastvideo/train/callbacks/grad_clip.py`: undocumented debug logging
        knobs/scaffolding removed.
    - Adjudicator validation:
      - Modal L40S app `ap-qQX6tc7AbGK9FQ9XbmcIoi`:
        `pytest tests/local_tests/tdm/ -v -s` passed,
        `13 passed in 17.67s`.
      - Modal L40S app `ap-kbQ8YHEfTCIETAoFVAuAlF`:
        `pre-commit run --files ...` passed for changed files.
    - Because the adjudicator changed code, Stage 3 must continue with a fresh
      `review-code` pass against the updated committed branch.
    - Spawned fresh read-only `review-code` sub-agent `Confucius`
      (`019f3ff4-f543-7d31-a7a7-36d8de98dc4f`) to review the updated branch
      after commit `8d46c5a7c`.
    - Reviewer found one actionable issue: `docs/training/train_infra.md`
      still says `noise_interval_mode` chooses a sampled larger sigma or
      terminal sigma, but commit `8d46c5a7c` changed `to_terminal` to avoid
      exact terminal sigma and instead target the highest trainable
      non-terminal sigma. The reviewer judged the terminal-mode trainability
      bug fixed, with remaining risk in public knob/docs accuracy.
    - Fresh independent adjudicator/fixer pass for that finding:
      - Verified `gh` identity as `macthecadillac`.
      - Re-read issue #775: open feature request for TDM; comments remain the
        maintainer interest comment and stale-bot comment.
      - Targeted open PR search for `775 OR TDM` returned `[]`; no PR state was
        changed.
      - Accepted the finding as valid. Code and tests after `8d46c5a7c`
        intentionally skip exact terminal `sigma=1.0` in `to_terminal`, while
        `docs/training/train_infra.md` still described terminal sigma as the
        target.
      - Implemented a narrow docs fix in `docs/training/train_infra.md`: the
        table now routes `noise_interval_mode` details to a note, and the note
        states that `separate` samples a larger non-terminal target while
        `to_terminal` skips exact terminal sigma because flow-SNR weighting
        gives it zero fake-score weight, using the highest trainable
        non-terminal target with a lower-sigma source instead.
      - Validation after this docs-only patch:
        `git diff --check` passed. Sandboxed changed-file pre-commit failed
        only because `uv` could not write to `/home/toolbox/.cache/uv`; the
        escalated rerun of
        `uvx pre-commit run --files docs/training/train_infra.md .agents/handoffs/issue-775-handoff.md`
        passed `codespell`, `PyMarkdown`, filename-space, and suggestion
        hooks, with Python hooks skipped because no Python files were changed.
        The prior Modal TDM unit validation at `8d46c5a7c` remains applicable
        for the trainability behavior because this patch does not change
        runtime code.
      - Final GitHub refresh before commit/push: issue #775 remains open,
        assigned to `macthecadillac`, and still has only the maintainer
        interest and stale-bot comments; targeted open PR search for
        `775 OR TDM` returned `[]`.
      - Committed and pushed signed commit
        `858845bba053130a626bc206498c6b7e5eedccea`
        `[docs]: clarify TDM terminal noise mode`. `git log -1
        --show-signature` verified a good signature from
        `Mac Lee <macthecadillac@gmail.com>` using subkey
        `9970C3F2BC145193A5C12AAD4C1D75FF3B58866D`.
    - Because the docs adjudicator changed the branch, Stage 3 continued with
      another fresh `review-code` pass.
    - Spawned fresh read-only `review-code` sub-agent `Dalton`
      (`019f4002-994b-7c81-8f3d-b4884062783f`) to review the updated branch
      after commit `858845bba053130a626bc206498c6b7e5eedccea`.
    - Reviewer found one actionable issue: `tdm_denoising_steps` accepts any
      non-empty list, but `_student_trajectory()` starts from pure random
      noise and therefore assumes the first configured timestep maps to the
      scheduler terminal / max sigma. If a user omits `1000`, reorders the
      list, or supplies duplicate/non-descending steps, training silently runs
      an off-distribution trajectory and target selection treats the max
      configured sigma as terminal.
    - Spawned fresh independent adjudicator/fixer sub-agent `Kuhn`
      (`019f400a-136c-7291-8544-26402e806916`) with only the issue/repo,
      committed branch/commit, and reviewer finding.
    - Adjudicator accepted the schedule-validation finding and rejected none.
    - Adjudicator implemented and pushed signed commit
      `5c2a8b9baf66b9d936cb565e2031fd02d8cdfacc`
      `[fix]: validate TDM denoising schedule`.
      `git log -1 --show-signature` verified a good signature from
      `Mac Lee <macthecadillac@gmail.com>` using subkey
      `9970C3F2BC145193A5C12AAD4C1D75FF3B58866D`.
    - Adjudicator changes:
      - `fastvideo/train/methods/distribution_matching/tdm.py`: validates at
        least two steps, terminal first sigma, and strictly decreasing mapped
        sigmas.
      - `tests/local_tests/tdm/test_tdm_method_unit.py`: added rejected
        bad-schedule cases.
      - `docs/training/train_infra.md` and
        `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`:
        documented the schedule contract.
    - Adjudicator validation:
      - Modal L40S app `ap-ijqla4arb5Rfpb5HZP4xMt`:
        `pytest tests/local_tests/tdm/ -q` passed,
        `17 passed in 21.96s`.
      - Modal L40S app `ap-BsuDShZm6exFvLDGQ1QlAz`:
        `pre-commit run --files ...changed files...` passed.
    - Because the adjudicator changed code/docs, Stage 3 must continue with a
      fresh `review-code` pass against the updated committed branch.
    - Spawned fresh read-only `review-code` sub-agent `Dirac`
      (`019f4013-4694-72d0-a4df-64057df2d5d2`) to review the updated branch
      after commit `5c2a8b9baf66b9d936cb565e2031fd02d8cdfacc`.
    - Reviewer reported **no actionable findings**. It confirmed the earlier
      terminal-mode, docs, stale README/debug scaffolding, and schedule
      validation findings appear resolved. Residual risk is validation depth:
      latest exact-SHA Modal TDM unit coverage and quality/convergence after
      the final schedule-validation changes.
    - Full pre-commit gate completed on Modal L40S app
      `ap-PbwmXNpbbB4s9MpK9PgyvW` at pushed commit
      `5c2a8b9baf66b9d936cb565e2031fd02d8cdfacc`:
      `pre-commit run --all-files` passed all hooks (`yapf`, `ruff`,
      `codespell`, `PyMarkdown`, `actionlint`, `mypy`, filename-space, and
      suggestion).
    - Stage 3 stopped after the no-actionable-findings review pass. No PR has
      been opened. The handoff remains active until explicit Stage 4 direction.
    - Remaining risk: post-final long-convergence visual quality is still
      unproven. The earlier checkpoint-500 validation outputs, generated
      before the final generator-target/terminal-mode/schedule-validation
      fixes, were visually degraded according to user review. The next quality
      spend should be a short post-final canary before any longer DGX run.
    - User requested an apples-to-apples visual test with the new final code:
      same amount of training as the degraded baseline, then the same student
      and teacher visual comparison outputs.
    - Refreshed GitHub state before launch: `gh` identity was
      `macthecadillac`; issue #775 remained open and assigned to
      `macthecadillac`; issue comments remained only the maintainer interest
      comment and stale-bot comment; targeted open PR search for `775 OR TDM`
      returned `[]`; no PR draft status was changed.
    - Launched a detached DGX Spark Docker run from branch head
      `49dda70a2be24a953fea0663f733a330b7a399ea`
      `[misc]: record TDM review completion`. This head includes final code
      commit `5c2a8b9baf66b9d936cb565e2031fd02d8cdfacc` plus the pushed
      review-completion handoff commit.
    - Run root:
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_49dda70_final_v1`.
      Container:
      `issue775_tdm_schedule_500_49dda70_final_v1_nonroot`, id
      `9dc80c5c537fbb888c335a685f8fa0ea82f860e4495d091379f922b0983f5d05`.
      It is launched detached with `docker run -d`, `--user 1006:1006`, and
      image `fastvideo-dev-nonroot:issue775`, the DGX-local non-root
      derivative of the required FastVideo Docker image. `docker top` verified
      wrapper, `tee`, `torchrun`, worker Python, and resource tracker all
      appear as user `mac`.
    - The run script clones `https://github.com/macthecadillac/FastVideo.git`,
      checks out exact commit
      `49dda70a2be24a953fea0663f733a330b7a399ea`, downloads
      `wlsaidhi/crush-smol_processed_t2v`, and runs the same 500-step 1-GPU
      Wan TDM interval-1 training shape as the degraded baseline:
      `max_train_steps=500`, `method.generator_update_interval=1`, JSONL
      tracking, checkpoints every `100` steps, and validation disabled during
      training with `callbacks.validation.every_steps=0`.
    - After training, the same run script is scheduled to generate checkpoint
      500 student videos from
      `/workspace/run/output/checkpoint-500` using the training validation
      callback with `sampling_steps=[4]`,
      `sampling_timesteps=[1000,750,500,250]`, `guidance_scale=6.0`, and output
      dir `/workspace/run/validation_step500/student_tdm_4step`.
      It then runs a standalone teacher/base Wan script for the same four
      prompts with `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `50` inference steps,
      `height=448`, `width=832`, `num_frames=77`, `fps=16`, and seed `1000`.
    - Expected host output paths after completion:
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_49dda70_final_v1/validation_step500/student_tdm_4step/validation_step_500_inference_steps_4_video_{0..3}.mp4`
      and
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_49dda70_final_v1/validation_step500/teacher_wan_50step/teacher_wan_50step_prompt{0..3}.mp4`.
      Prompt mapping will be saved at
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_49dda70_final_v1/validation_step500/teacher_wan_50step/prompts.json`.
    - Startup status checks:
      `docker inspect` reports
      `status=running running=true exit=0 oom=false`, started
      `2026-07-08T06:35:06.698493347Z`. The log entered
      `training_to_500` at `2026-07-08T06:35:33+00:00` and applied the
      expected CLI overrides. No tracker rows or checkpoints existed at the
      last startup poll because the process was still downloading/loading the
      Wan model snapshot; cache size had reached `13G` with active incomplete
      HF/Xet shard files. Disk had `2.8T` free. Leave the detached container
      running unless the user asks to stop or relaunch it.
    - Follow-up training poll: the same container remained
      `status=running running=true exit=0 oom=false` and had reached
      step `10/500`. Tracker summary from
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_49dda70_final_v1/output/tracker/metrics.jsonl`:
      `40` JSONL rows, steps `1..10`, `10` loss rows, `0` nonfinite scalar
      metrics. Generator diagnostics sampled timesteps `[500.0, 750.0]`,
      confirming the final code no longer samples generator guidance from the
      earlier off-support `1000` or `250` endpoints. Last loss row at step
      `10`: `total_loss=0.3162887692451477`,
      `generator_loss=0.31520676612854004`,
      `fake_score_loss=0.001082007889635861`,
      `tdm/generator/timestep=500.0`, `tdm/generator/sigma=0.5`, and
      `step_time_sec=106.68669968913309`.
    - Completion status checked on 2026-07-09: container
      `issue775_tdm_schedule_500_49dda70_final_v1_nonroot` reports
      `status=exited running=false exit=0 oom=false`, started
      `2026-07-08T06:35:06.698493347Z`, finished
      `2026-07-08T22:24:09.613773078Z`.
    - Completed-run tracker summary:
      `2000` JSONL rows, `500` unique steps, steps `1..500`, `500` loss rows,
      `0` nonfinite scalar metrics, and generator timesteps `[500.0, 750.0]`.
      Last loss row at step `500`: `total_loss=0.3945949673652649`,
      `generator_loss=0.39244306087493896`,
      `fake_score_loss=0.002151909749954939`,
      `tdm/generator/timestep=750.0`, and `tdm/generator/sigma=0.75`.
    - Checkpoints present:
      `checkpoint-100`, `checkpoint-200`, `checkpoint-300`, `checkpoint-400`,
      and `checkpoint-500`.
    - Final visual outputs were generated successfully:
      student checkpoint-500 4-step videos at
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_49dda70_final_v1/validation_step500/student_tdm_4step/validation_step_500_inference_steps_4_video_{0..3}.mp4`
      with file sizes approximately `246K`, `1.1M`, `139K`, and `1.3M`;
      teacher/base Wan 50-step videos at
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_49dda70_final_v1/validation_step500/teacher_wan_50step/teacher_wan_50step_prompt{0..3}.mp4`
      with file sizes approximately `324K`, `852K`, `247K`, and `1.1M`.
      Prompt mapping is saved at
      `/home/mac/fastvideo-runs/issue-775/tdm_schedule_500_49dda70_final_v1/validation_step500/teacher_wan_50step/prompts.json`.
    - `ffprobe` sanity check passed for all eight MP4s: each reports
      `832x448`, `77` frames, `16` fps, and `4.812500s` duration.
      The run log ended with `=== done 2026-07-08T22:24:09+00:00 ===`.

- Resumed at `2026-07-06 04:04:09 UTC` from
  `/tmp/fastvideo-worktrees/issue-775-tdm`.
- Verified `gh` identity outside the sandbox: `macthecadillac`.
- Stage resumed: **Stage 2 - Implement The User-Directed Fix**.
- Refreshed issue state with `gh issue view 775 -R hao-ai-lab/FastVideo`:
  issue remains open, assigned to `macthecadillac`, updated
  `2026-06-30T13:58:39Z`, with no comments beyond `zhisbug`'s maintainer
  interest comment and the stale-bot comment already recorded below.
- Refreshed open PR state with `gh pr list` and narrowed related search:
  no open PR explicitly references issue `775` or `TDM` in title, head branch,
  body, or closing issue references. No PR draft status was changed.
- Current next action remains the validation/diagnostic decision recorded
  below: either run the full original interval-1 resume-to-200 with
  checkpoint saves and validation enabled, or decide to stop/redirect the
  quality investigation before Stage 3.
- Modal volume preflight with
  `uvx modal volume ls hf-model-weights tdm_pilot_sde_200_interval1_35888898_dataset`
  confirmed only `checkpoint-100`, `tracker`, and step-0/step-100 validation
  videos are present. No `checkpoint-200` or step-200 validation artifacts
  existed before launching the next resume attempt.
- First full resume launch attempt app `ap-gQAnwHSK5VFBaclkNhVUhI` failed
  before training. Modal cloned `https://github.com/hao-ai-lab/FastVideo.git`
  and could not check out `e0fd865365f6d92d1e9bbef0dc604f303030cc2e`
  (`fatal: reference is not a tree`) because the issue branch commit is on
  fork remote `git@github.com:macthecadillac/FastVideo.git`, branch
  `issue/775-tdm`. Rerun should use
  `https://github.com/macthecadillac/FastVideo.git` for the Modal clone.
- Corrected full resume launch app `ap-1ejJtcQ4y3iUZ4JSBuT0Zz` started from
  `https://github.com/macthecadillac/FastVideo.git` at commit
  `e0fd865365f6d92d1e9bbef0dc604f303030cc2e`. Checkout succeeded. Command
  resumes from
  `/root/data/tdm_pilot_sde_200_interval1_35888898_dataset/checkpoint-100`,
  runs to `max_train_steps=200`, sets
  `method.generator_update_interval=1`, keeps checkpoint saves every `100`
  steps, enables validation every `100` steps with
  `offload_training_state=true` and `unload_pipeline_after_validation=true`,
  uses JSONL tracking only, and commits the Modal volume.
- Early stream progress: checkpoint loaded and resumed from step `100`;
  resume-time validation completed, optimizer/teacher/critic state restored,
  RNG snapshot restored, and resumed training completed through step `104`.
  Steps `101..104` each logged student grad, critic grad, loss metrics with
  `update_student`, and EMA rows.
- Continued stream progress: completed through step `110` with the same full
  JSONL row set for every resumed step. Progress-bar step time stabilized at
  about `40.8..42.0s` per step.
- Continued stream progress: completed through step `130`; every streamed
  step still logged student grad, critic grad, loss metrics with
  `update_student`, and EMA rows. No hang or nonfinite metric was visible in
  the stream.
- Mid-run stream progress: completed through step `150` with the same full
  JSONL row set. Step times remain about `40..41s`; no visible stalled step,
  missing row group, or process error.
- Later stream progress: completed through step `170` with student grad,
  critic grad, loss metrics, and EMA rows for every streamed step. No visible
  stalled step, missing row group, process error, or nonfinite warning.
- Later stream progress: completed through step `180` with all expected
  tracker row groups. The run is now approaching the previous stopped region
  around steps `193..194`.
- Critical region cleared in the live stream: step `193` logged student grad,
  critic grad, loss metrics, and EMA rows, and step `194` also completed all
  four row groups. This rerun does not reproduce a deterministic full-command
  hang between steps `193` and `194`.
- Full interval-1 resume result: app `ap-1ejJtcQ4y3iUZ4JSBuT0Zz` completed
  successfully, reached step `200`, saved `checkpoint-200`, ran step-200
  validation, committed the Modal volume, and exited with code `0`. Non-fatal
  warnings matched known behavior: duplicate final `checkpoint-200` overwrite
  warnings and NCCL `destroy_process_group()` shutdown warnings.
- Modal volume after completion contains `checkpoint-100`, `checkpoint-200`,
  `tracker`, and validation videos for steps `0`, `100`, and `200`.
- Downloaded lightweight artifacts to
  `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_pilot_sde_200_interval1_full_resume/`:
  tracker `metrics.jsonl`, `artifacts.jsonl`, `config.json`,
  `tracker/files/run.yaml`, all four step-200 MP4s, frame-38 PNG for prompt 0,
  and contact sheet
  `prompt0_frame38_interval1_contact_sheet.png`.
- De-duplicated tracker summary after keeping the latest row per
  `(step, row_group)`:
  - rows: `800`; steps: `1..200` (`200` unique).
  - row groups: `200` student-grad, `200` critic-grad, `200` loss, `200` EMA.
  - missing/nonunique row groups after de-duplication: `0`.
  - nonfinite scalar metrics: `0`.
  - `update_student` is `1.0` for all `200` loss rows.
  - `tdm/fake_score/sigma_to_is_terminal` is `0.0` for all `200` loss rows.
  - `tdm/fake_score/snr_weight` is `1.0` for all `200` loss rows.
  - `total_loss`: min `0.1882624775`, max `0.9327092767`, mean
    `0.4572506941`, first `0.3315637112`, last `0.4248001873`.
  - `fake_score_loss`: min `0.0003644582`, max `0.0016866034`, mean
    `0.0010596468`, first `0.0015967983`, last `0.0011549045`.
  - `generator_loss`: min `0.1868882924`, max `0.9316013455`, mean
    `0.4561910472`, first `0.3299669027`, last `0.4236452878`.
  - `step_time_sec`: min `39.8365791600`, max `51.9692875260`, mean
    `40.6320101186`, first `51.2910224430`, last `40.5539538170`.
  - Steps `193`, `194`, and `200` each have exactly one student-grad,
    critic-grad, loss, and EMA row after de-duplication.
- `ffprobe` sanity check on each step-200 MP4: `832x448`, `77` frames,
  `16` fps, duration `4.812500s`.
- Visual comparison for prompt 0 frame 38:
  teacher 50-step is clear and prompt-conditioned; base Wan 4-step DMD, old
  interval-5 TDM step-200, and new interval-1 TDM step-200 all remain heavily
  blurred/frosted with no clear prompt-conditioned object. The interval-1
  full resume proves the run can complete and produce artifacts, but it does
  not show visible 4-step quality improvement at step 200.
- Current next decision: either spend a longer interval-1 convergence pilot to
  see whether more student updates eventually improve 4-step quality, or stop
  convergence spending for now and debug objective/sigma schedule/weighting
  before additional long runs. Given the completed 200-step interval-1 output
  is still visibly blurred, the recommended next direction is objective /
  schedule / weighting investigation rather than immediately scaling the same
  settings.

## Fix-Issue Resume State

- Handoff path:
  `.agents/handoffs/issue-775-handoff.md`
- Current `fix-issue` stage:
  **Stage 3 - Review, Adjudicate, And Iterate**, completed and awaiting user
  decision on whether to proceed to Stage 4 draft PR creation.
- Implementation has begun:
  yes. The branch contains TDM implementation, tests/docs, JSONL tracker
  support, and cleanup from the review/adjudication loop.
- Stage 1 status:
  complete. The user approved implementation earlier in the thread.
- Stage 2 status:
  complete for the current user-directed implementation. The latest code
  commit is `5c2a8b9baf66b9d936cb565e2031fd02d8cdfacc`.
- Stage 3 status:
  complete. Review/adjudication loop ended with no actionable findings after
  review sub-agent `Dirac`. Full Modal pre-commit passed at
  `5c2a8b9baf66b9d936cb565e2031fd02d8cdfacc`.
- Stage 4 status:
  not started. No PR exists for this branch; do not remove this handoff until
  immediately before an explicitly requested draft PR creation.
- Next resume action:
  if the user asks to open the draft PR, run the Stage 4 pre-PR gate, transfer
  final handoff context into the PR body, remove this handoff with `git rm`,
  commit/push that removal, and create only a draft PR. If the user instead
  asks for more quality validation, run a short post-final canary before any
  longer DGX convergence run.

## Worktree

- Repo: `github.com/hao-ai-lab/FastVideo`
- Branch: `issue/775-tdm`
- Worktree: `/tmp/fastvideo-worktrees/issue-775-tdm`
- Modal launcher worktree: `/tmp/fastvideo-worktrees/interleavethinker-modal`
  on branch `interleavethinker`
- Handoff: `.agents/handoffs/issue-775-handoff.md`
- Latest pushed issue-branch commit before this handoff update:
  `5c2a8b9baf66b9d936cb565e2031fd02d8cdfacc`
  `[fix]: validate TDM denoising schedule`

## Draft PR Message

Title: `[feat]: add Wan TDM training method`

```markdown
Fixes #775

## Summary

- Add a Wan-native Trajectory Distribution Matching training method for the
  modular trainer, with student/teacher/critic roles, Wan flow noising, TDM
  trajectory sampling, fake-score training, and generator score diagnostics.
- Add a Wan 2.1 T2V 1.3B LoRA TDM example config plus training docs and focused
  local TDM tests.
- Add JSONL tracker support for durable scalar/artifact diagnostics.
- Align generator score targets with fake-score target support, keep terminal
  mode trainable by avoiding exact zero-SNR terminal sigma, and validate TDM
  denoising schedules before training.

## Validation

- Modal L40S `pytest tests/local_tests/tdm/ -v -s`
  - `12 passed in 20.86s`
  - App: `ap-dkepDHPikkLfo8jyen37kB`
- Modal H100 two-step Wan TDM real-training smoke
  - App: `ap-v4R2pPiPPxl4zIC6FCoMnP`
  - JSONL summary: 8 rows, 2 loss rows, steps `[1, 2]`, generator timesteps
    `[500.0, 500.0]`, generator sigmas `[0.5, 0.5]`, 0 nonfinite scalar metrics
- Modal L40S post-adjudication TDM unit validation
  - `pytest tests/local_tests/tdm/ -q`
  - `17 passed in 21.96s`
  - App: `ap-ijqla4arb5Rfpb5HZP4xMt`
- Modal L40S changed-file pre-commit after schedule validation
  - passed
  - App: `ap-BsuDShZm6exFvLDGQ1QlAz`
- Modal L40S full pre-commit at `5c2a8b9baf66b9d936cb565e2031fd02d8cdfacc`
  - `pre-commit run --all-files`
  - passed
  - App: `ap-PbwmXNpbbB4s9MpK9PgyvW`
- DGX Spark 500-step interval-1 diagnostic run completed before the final
  schedule fixes; it produced checkpoints through `checkpoint-500` and finite
  metrics, but checkpoint-500 visual validation was heavily degraded.

## Review Loop

- Review pass 1 found terminal-mode zero-weight critic training, stale TDM test
  README content, and leftover grad-clip debug scaffolding. The adjudicator
  accepted all three and pushed `8d46c5a7c`.
- Review pass 2 found stale `to_terminal` docs. The adjudicator accepted it
  and pushed `858845bb`.
- Review pass 3 found missing validation for malformed TDM denoising schedules.
  The adjudicator accepted it and pushed `5c2a8b9b`.
- Final review pass reported no actionable findings.

## GPU Memory Impact

This adds a three-role TDM training method using trainable student/critic LoRA
modules plus a frozen teacher, so TDM training is inherently heavier than
single-model fine-tuning. The example config keeps the first target LoRA-only
and uses existing validation offload controls. No inference pipeline memory
increase is expected.

## Remaining Risk

Post-final long-convergence quality is not yet proven. Earlier checkpoint-500
videos, generated before the final generator-target/terminal-mode/schedule
fixes, were visibly degraded. The next quality step should be a short
post-final canary before spending on another long DGX run.

# Checklist
- [x] I ran pre-commit run --all-files and fixed all issues
- [x] I added or updated tests for my changes
- [x] I updated documentation if needed
- [x] I considered GPU memory impact of my changes
For model/pipeline changes, also check:
- [ ] I verified targeted Wan T2V SSIM regression tests pass on L40S
- [ ] I updated the support matrix if adding a new model
```

## GitHub State

Last checked: 2026-07-06 with `gh api user --jq .login`; result:
`macthecadillac`.

- Issue: https://github.com/hao-ai-lab/FastVideo/issues/775
- Title: `[Feature] TDM`
- State: open
- Created: 2025-09-01T08:20:12Z
- Updated: 2026-06-30T13:58:39Z
- Author: `fenght96`
- Assignee: `macthecadillac`
- Labels: `good first issue`, `contribution-needed`, `stale`, `keep-open`
- Body asks whether FastVideo has a plan for TDM and links the paper.
- Comments reviewed:
  - 2026-02-04 maintainer interest comment from `zhisbug`.
  - 2026-05-31 stale-bot comment.
- Open PR state:
  full open PR list was refreshed on 2026-07-06; targeted open PR searches for
  `775` and `TDM` returned `[]`. No active PR directly targets TDM or issue
  775, and no PR draft status was changed.

## Scope Decision

- Treat this as a **Wan-native TDM training-method integration**, not an
  `/add-model` model/component port.
- First target: Wan 2.1 T2V 1.3B, text-only/data-free, LoRA-first.
- Production code must not import from `Luo-Yihong/TDM`.
- Keep official CogVideoX LoRA import and TSAM/multi-NFE support out of the
  first scope.

## Reference Summary

Reference sources read:

- Paper: `Learning Few-Step Diffusion Models by Trajectory Distribution
  Matching`, arXiv:2503.06674
- Project: https://tdm-t2x.github.io/
- Repo: https://github.com/Luo-Yihong/TDM
- Script: `train_tdm_demo.py`

Important reference behavior:

- Roles are trainable generator/student, frozen teacher, trainable fake-score
  model.
- Generated trajectory is built under `torch.no_grad()`.
- Generator update samples from that no-grad trajectory and recomputes one
  student prediction for the generator loss.
- No reference equivalent exists for a user-facing full-rollout gradient
  option.
- The reference is diffusion/DDPM-style; FastVideo Wan uses flow matching.

## Implementation State

Key files:

- `fastvideo/train/methods/distribution_matching/tdm.py`
- `fastvideo/train/methods/distribution_matching/__init__.py`
- `fastvideo/train/methods/__init__.py`
- `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`
- `tests/local_tests/tdm/`
- `docs/training/train_infra.md`

Implemented behavior:

- `TDMMethod` subclasses `DMD2Method` to reuse role validation, optimizers,
  fake-score optimizer plumbing, CFG parsing, and backward routing.
- Required roles: `student`, `teacher`, `critic`.
- Required rollout mode: `simulate`.
- Flow bridge uses Wan noising:

  ```text
  x_sigma = (1 - sigma) * x0 + sigma * eps
  ```

- Between-sigma transition helper:

  ```text
  a = (1 - sigma_to) / (1 - sigma_from)
  beta_sq = sigma_to^2 - (a * sigma_from)^2
  x_to = a * x_from + sqrt(beta_sq) * eps_rand
  mixed_eps = (a * sigma_from * eps_from + sqrt(beta_sq) * eps_rand) / sigma_to
  ```

- `noise_interval_mode: separate` samples strictly larger non-terminal target
  sigmas only.
- `noise_interval_mode: to_terminal` remains explicit terminal behavior.
- `student_sample_type` supports `sde` and `ode`; first config stays `sde`.
- TDM no longer exposes `enable_gradient_in_rollout`; reference-style
  final-prediction-only gradients are unconditional.
- JSONL tracker support exists for persisted scalar diagnostics.

## Key Commits

- `0ebbcbf5` `[feat]: add JSONL training tracker`
- `ca654254` `[fix]: avoid terminal sigma in TDM separate noising`
- `1afc867c` `[fix]: remove TDM rollout gradient option`
- `008883d9` `[misc]: record TDM ode rollout diagnostic`
- `4e7d9e76` `[misc]: compact TDM handoff and pilot plan`
- `ea7e8704` `[misc]: record TDM 200-step pilot`
- `13d76da9` `[misc]: record TDM teacher comparison artifact`
- `a0a4f93a` `[misc]: record TDM visual review`
- `e574a636` `[debug]: instrument gradient clipping for TDM hang`
- `3550048b` `[debug]: log grad clip diagnostics from all ranks`
- `9fdd20cb` `[debug]: record interval-1 control diagnostic`
- `946ddcad` `[debug]: update validation repro progress`
- `9e2dc9aa` `[debug]: record interval validation repro result`

## Validation Summary

Local machine:

- Do not run pytest locally. Repo guidance says validation should run on Modal.
- Local syntax/type-oriented checks used so far:
  - `python -m py_compile ...`
  - `git diff --check`
  - `uvx pre-commit run --files ...`
  - direct `uvx mypy ... --explicit-package-bases ...`
- Pre-commit's mypy hook fails before type-checking in this worktree because
  `issue-775-tdm` is not a valid Python package name. Direct mypy passes.

Modal smoke/unit evidence:

- App `ap-QFB1GdYJkjQDHriHhY9Oor`
- Commit `1afc867c`
- Command: `pytest tests/local_tests/tdm/ -v -s`
- Result: `10 passed, 14 warnings`

20-step SDE diagnostic:

- App `ap-C7T7LCY4j0iGdCUigW85XX`
- Output root: `/root/data/tdm_diag_fake_score_fix_20_ca654254`
- Metrics: `/tmp/tdm_diag_fake_score_fix_20_ca654254_metrics.jsonl`
- Result: completed on 4x L40S with finite losses.
- No terminal fake-score targets, zero SNR weights, zero final weights, zero
  fake-score losses, or nonfinite scalar metrics.
- `update_student`: `[5, 10, 15, 20]`

20-step ODE diagnostic:

- App `ap-x6X9P0DJPpdV5LB5LIygWM`
- Output root: `/root/data/tdm_diag_ode_20_b55aeb32`
- Metrics: `/tmp/tdm_diag_ode_20_b55aeb32_metrics.jsonl`
- Result: completed on 4x L40S with finite losses.
- No terminal fake-score targets, zero SNR weights, zero final weights, zero
  fake-score losses, or nonfinite scalar metrics.
- ODE mean step time was lower than SDE in the 20-step diagnostic, but this is
  not quality evidence.

Current config decision:

- Keep `student_sample_type: sde` as first default because it more closely
  matches the upstream reference, which re-noises each predicted clean latent
  with model-predicted noise at the next timestep.
- Keep ODE as a viable alternative pending longer/qualitative evidence.

200-step SDE pilot:

- App: `ap-jDoLULWN2Y6VACZJCHlCQr`
- Commit: `4e7d9e760df04c9e569a47215e74e84bbdf6a7dd`
- Output root: `/root/data/tdm_pilot_sde_200_4e7d9e76`
- Command summary: 4x L40S, Wan TDM default config, `student_sample_type: sde`,
  `max_train_steps=200`, JSONL tracker enabled, validation every 100 steps,
  DCP training-state checkpoints every 100 steps, `--commit-volume`.
- Result: completed successfully. Training progress reached `200/200` in
  about `1:17:55`; Modal volume commit completed.
- Non-fatal warnings observed:
  - `checkpoint-200` was saved at the step-200 checkpoint boundary and then
    saved again by final training teardown, producing PyTorch DCP overwrite
    warnings.
  - NCCL warned that `destroy_process_group()` was not called before process
    exit. The job still exited with code 0.
- Metrics file:
  `/tmp/tdm_pilot_sde_200_4e7d9e76_metrics.jsonl` downloaded from
  `/root/data/tdm_pilot_sde_200_4e7d9e76/tracker/metrics.jsonl`.
- Metrics summary:
  - JSONL rows: `640`
  - Unique steps: `1..200`
  - Loss rows: `200`
  - Critic grad rows: `200`
  - Student grad rows: `40`
  - `update_student` steps exactly matched `[5, 10, ..., 200]`
  - Nonfinite scalar metrics: `0`
  - `sigma_to_is_terminal` steps: `0`
  - zero `snr_weight` steps: `0`
  - zero `weight_mean` steps: `0`
  - zero `fake_score_loss` steps: `0`
  - `total_loss`: min `0.0004079751`, max `0.7137516737`,
    mean `0.0966718541`, first `0.0016110574`, last `0.6687002182`
  - `fake_score_loss`: min `0.0004079751`, max `0.0017997533`,
    mean `0.0010606790`, first `0.0016110574`, last `0.0012961911`
  - `generator_loss`: min `0.0`, max `0.7124048471`,
    mean `0.0956111750`, first `0.0`, last `0.6674040556`
  - `step_time_sec`: min `18.0589167890`, max `42.2056847560`,
    mean `23.0179435935`
- Tracker artifacts exist:
  - `tracker/config.json`
  - `tracker/metrics.jsonl`
  - `tracker/artifacts.jsonl`
  - `tracker/files`
- Checkpoints exist:
  - `checkpoint-100/{metadata.json,rng_state_rank0.pt,...,rng_state_rank3.pt,dcp}`
  - `checkpoint-200/{metadata.json,rng_state_rank0.pt,...,rng_state_rank3.pt,dcp}`
  - each DCP directory contains `.metadata` plus `__0_0.distcp` through
    `__3_0.distcp`.
- Validation artifacts:
  - 12 MP4s exist: four each at steps `0`, `100`, and `200`.
  - Downloaded to
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_pilot_sde_200_4e7d9e76_validation_named/`
    after the 2026-07-02 restart wiped `/tmp`.
  - All 12 decoded with temporary `imageio`/`imageio-ffmpeg` tooling as
    `(77, 448, 832, 3)` `uint8`.
  - All basic non-blank checks passed: pixel std range
    `11.896296..39.797680`; first-vs-last frame absolute mean delta range
    `1.095667..4.911918`.

Prompt 0 teacher/student comparison:

- User asked for both the Wan teacher output and TDM student outputs for one
  prompt, plus the prompt text, for visual inspection.
- Chosen prompt index: `0` from
  `examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json`.
- Prompt text:

  ```text
  In the video, a woman is elegantly showcasing her earrings, bringing attention to their intricate design with a gentle touch of her fingers. She is bathed in ambient purple and pink lighting, which casts a soft glow on her delicate features and enhances the vivid tones of her lipstick and eye makeup. Her hair is styled to frame her face smoothly, emphasizing the contours of her jawline and cheekbones. The background features a blurred neon light, adding an artistic and modern touch to the overall aesthetic.
  ```

- Teacher generation Modal app: `ap-wVSQpGeoOvGexEGqkzn4hv`
- Teacher output root in Modal volume:
  `/root/data/issue_775_tdm_prompt0_compare`
- Teacher settings:
  - Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
  - Pipeline: normal `WanPipeline`
  - Inference steps: `50`
  - Seed: `1000`
  - Resolution: `448x832`
  - Frames/FPS: `77` / `16`
  - Guidance scale: `6.0`
  - Flow shift: `8.0`
  - Logged generation time: `216.996s`; e2e latency `219.733s`
  - Peak memory: `14352.989 MB`
- Local assembled comparison folder:
  `/home/toolbox/FastVideo/outputs/issue-775-tdm/prompt0_teacher_student_compare/`
- Files in the comparison folder:
  - `teacher/issue_775_tdm_prompt0_compare/wan_teacher_50_steps_seed1000_prompt0.mp4`
  - `teacher/issue_775_tdm_prompt0_compare/wan_teacher_50_steps_seed1000_prompt0.json`
  - `teacher/issue_775_tdm_prompt0_compare/prompt0.txt`
  - `tdm_student_4step/tdm_student_step000_4steps_seed1000_prompt0.mp4`
  - `tdm_student_4step/tdm_student_step100_4steps_seed1000_prompt0.mp4`
  - `tdm_student_4step/tdm_student_step200_4steps_seed1000_prompt0.mp4`
  - `README.md`
- Note: these are qualitative artifacts. The teacher and TDM student use
  different sampler trajectories, so they are not expected to be pixel-matched.
  Local `ffprobe` failed to initialize the H.264 decoder in this environment,
  but the Modal log confirms the teacher MP4 was generated, decoded, and saved.
- User visual assessment:
  - Teacher output is imperfect but clearly conditioned by the prompt, vivid,
    and detailed.
  - TDM student output does not get visibly worse from `step0` to `step100`
    to `step200`.
  - TDM student output remains extremely blurred at all inspected steps, with
    no identifiable objects; user described it as if the teacher output were
    seen through heavily frosted glass.
- Interpretation:
  - This does not prove numerical failure because an untrained/base Wan model
    sampled with the aggressive 4-step DMD schedule can be very blurry, and the
    200-step pilot contains only about 40 student/generator updates.
  - It does mean the pilot has not shown qualitative learning yet. Before
    spending on a much longer run, first distinguish expected 4-step base blur
    from an integration/validation issue.

Prompt 0 targeted blur diagnostics:

- Base Wan 4-step DMD baseline:
  - First attempt app `ap-AD8N2PjcdD064G53krRQZb` failed before model code
    because Modal could not check out newest handoff-only commit `a0a4f93a`
    (`fatal: reference is not a tree`). Code diagnostics therefore used
    known-visible code commit `ea7e8704`; production code was unchanged between
    those commits.
  - Successful app: `ap-w5GquHcspYVAYi3QadKluL`
  - Commit: `ea7e8704502aaa0485aa9dba921416c66c1542b6`
  - Pipeline: `WanDMDPipeline`
  - Settings: same prompt, seed `1000`, `448x832`, `77` frames, `fps=16`,
    `guidance_scale=6.0`, `embedded_cfg_scale=6.0`, `flow_shift=8.0`,
    `num_inference_steps=4`,
    `dmd_denoising_steps=[1000,750,500,250]`.
  - Modal output:
    `/root/data/issue_775_tdm_prompt0_compare/base_dmd_4step/base_wan_dmd_4steps_seed1000_prompt0.mp4`
  - Local output:
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/prompt0_teacher_student_compare/base_dmd_4step/base_dmd_4step/base_wan_dmd_4steps_seed1000_prompt0.mp4`
  - Visual/stat result: base Wan 4-step DMD is also heavily blurred.
- DCP export / normal inference loader probe:
  - App: `ap-WxB3oDIhzdiRT8PCK7DjRZ`
  - `dcp_to_diffusers` successfully exported the step-200 student transformer
    to
    `/root/data/issue_775_tdm_prompt0_compare/student_step200_diffusers`,
    but normal `VideoGenerator.from_pretrained(export_path, ...)` failed
    because `model_index.json` was a broken symlink. Probe app
    `ap-jyjjWsfqEaM9Q9IZOqbxEx` confirmed `Path.exists() == False` for that
    symlink even though the name appeared in directory listings. Root cause:
    the converter copies HF snapshot files with `symlinks=True`, preserving
    links to blobs outside the export root.
  - A follow-up attempt to load the exported student transformer through
    `init_weights_from_safetensors` failed in app `ap-3gQijLcOOHmKJrx251E0IV`
    with missing normal-loader key
    `blocks.0.attn2.to_k.base_layer.bias`. The exported state still contains
    LoRA-wrapper key structure and is not directly loadable by standard
    inference without a LoRA merge/export path.
- Live training-module high-step validation:
  - Temporary diagnostic script:
    `scripts/diagnostics/tdm_resume_validate.py`, applied to Modal with
    `--apply-local-patch`; this script was only for the diagnostic and is not
    intended to remain in the final branch.
  - App: `ap-Tqz1iA99t1AO6Lwr7ztDpE`
  - Commit: `ea7e8704502aaa0485aa9dba921416c66c1542b6` plus the temporary
    local diagnostic script patch.
  - Command:
    `torchrun --standalone --nproc_per_node=4 scripts/diagnostics/tdm_resume_validate.py --checkpoint /root/data/tdm_pilot_sde_200_4e7d9e76/checkpoint-200 --output-dir /root/data/issue_775_tdm_prompt0_compare/student_step200_live_wan_50step --inference-steps 50`
  - Result: completed successfully on 4x L40S. The checkpoint resumed at step
    `200`, EMA callback initialized/restored, validation offloaded optimizer
    state plus teacher/critic transformers, and the normal `WanPipeline` used
    the already-provided live student transformer for a 50-step sample.
  - Modal output:
    `/root/data/issue_775_tdm_prompt0_compare/student_step200_live_wan_50step`
  - Local output:
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/prompt0_teacher_student_compare/student_step200_live_wan_50step/`
  - Downloaded MP4s:
    `validation/validation_step_200_inference_steps_50_video_0.mp4` through
    `video_3.mp4`. The four copies are expected because the single validation
    prompt was padded across four SP groups; use `video_0` as the canonical
    comparison file.
  - OpenCV metadata for `video_0`: opened successfully, `77` frames, `16`
    fps, `832x448`. First-frame stats: mean `84.0549`, std `88.2849`.
  - Comparison stats from the same OpenCV probe:
    - Teacher 50-step first frame: mean `85.0922`, std `88.6851`
    - TDM step-200 4-step first frame: mean `63.6605`, std `16.8114`
    - Base Wan DMD 4-step first frame: mean `63.4405`, std `16.6299`
  - Representative contact sheet:
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/prompt0_teacher_student_compare/thumbnails/prompt0_frame38_contact_sheet.png`
  - Visual result: teacher 50-step and live step-200 student sampled at 50
    steps are both vivid and prompt-conditioned. Base Wan 4-step DMD and TDM
    step-200 4-step are both blurred/frosted.
- Current interpretation:
  - The step-200 student checkpoint and live validation/EMA application are
    not globally broken; when sampled with normal 50-step Wan, the live
    student still produces a clear prompt-conditioned video.
  - The observed failure is concentrated in the aggressive 4-step DMD/TDM
    trajectory. The 200-step pilot did not visibly improve over the blurred
    base 4-step DMD baseline, which is plausible with only about 40 generator
    updates but is not convergence evidence.
  - Separate cleanup is needed for `dcp_to_diffusers`: copied HF snapshot
    symlinks should be materialized, and exported LoRA-wrapped student weights
    need a merge/export path before normal inference can load them.

Latest quality assessment from user-visible artifacts:

- The inspectable 4-step videos available locally still look more or less the
  same as earlier outputs.
- Honest conclusion: the short 200-step interval-5 pilot did not produce a
  visible 4-step quality improvement. This does not prove TDM cannot work on
  Wan, but it means the current training budget/settings have not yet moved
  visible 4-step quality.
- Important caveat: the later interval-1 jobs from checkpoint-100 were
  debugging runs. They completed controlled runs to step 195, but checkpoint
  saving was disabled and they did not produce new inspectable step-195 or
  step-200 videos.
- The next meaningful quality artifact requires an interval-1 run that saves
  and validates at step 200, or a longer convergence pilot after that path is
  shown stable.

## Modal Data And Weights

Wan weights in Modal volume:

```text
/root/data/.cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
```

Small staged preprocessed T2V dataset:

```text
/root/data/.cache/datasets--wlsaidhi--crush-smol_processed_t2v/snapshots/67dd07f2163ad2b3397f8b3d8125b67ca452dc85/combined_parquet_dataset
```

Dataset summary from earlier Modal runs:

- 4 parquet shards
- 32 rows
- 8 rows per SP group in 4-GPU runs

## Completed Pilot Checks

- Job completion: passed.
- Metrics JSONL persisted with 200 loss rows: passed.
- Finite `total_loss`, `fake_score_loss`, and `generator_loss`: passed.
- `fake_score_loss` did not collapse to all zeros: passed.
- `update_student` followed `generator_update_interval=5`: passed.
- Terminal-sigma diagnostics stayed clean: passed.
- Validation MP4s exist at expected steps and decode successfully: passed.
- Video frame count and resolution match the Wan validation config: passed.
- Pixel statistics are not constant/all-black/all-white/corrupt: passed.
- DCP checkpoint directories exist for expected saved steps: passed.
- Base Wan 4-step DMD baseline for prompt 0 was generated: passed.
- Live step-200 student resumed and sampled with normal 50-step Wan validation:
  passed.
- Live 50-step student output is clear/prompt-conditioned while 4-step base and
  4-step TDM outputs are blurred: passed.

## Interval-1 Diagnostic State

Primary interval-1 student-update diagnostic for the same Wan TDM LoRA setup:

- Rationale: the 200-step SDE pilot used `generator_update_interval=5`, so it
  produced only about 40 student/generator updates. The critic was healthy and
  validation/EMA wiring is now proven, but the 4-step student output did not
  improve over the blurred base 4-step DMD baseline.
- First diagnostic to run: `max_train_steps=200` with
  `method.generator_update_interval=1`. This gives about 200 student updates,
  matching the student-update count of a `1000` outer-step run with interval
  `5`, but with fewer critic-only outer steps.
- Keep all other meaningful settings aligned with the current default TDM
  config: SDE rollout, LoRA rank/alpha, Wan 2.1 T2V 1.3B, 4-step validation
  schedule `[1000,750,500,250]`, seed `1000`, `448x832`, `77` frames.
- Expected validation targets: 4-step videos at steps `0`, `100`, and `200`,
  metrics JSONL, and DCP checkpoints. Compare prompt 0 against the existing
  teacher 50-step, base Wan DMD 4-step, and interval-5 200-step TDM artifacts.
- Decision rule:
  - If interval `1` stays stable and improves 4-step contrast/detail, consider
    changing the recommended TDM diagnostic/default cadence or running a
    longer interval-1 pilot.
  - If interval `1` stays stable but remains blurred, prioritize objective /
    sigma schedule / weighting debugging over simply increasing outer steps.
  - If interval `1` is unstable, keep interval `5` and run the longer 1000-step
    convergence pilot as originally planned.

Interval-1 run status:

- First launch app `ap-ZS7yR5ozq2vlVb8ruIOmox` failed before training because
  the command missed the Modal dataset override and used the YAML placeholder
  `data/Wan-Syn_77x448x832_600k`. Root error:
  `FileNotFoundError: No parquet files found under dataset path`.
- Corrected launch app: `ap-vX9fvoypEPLOOXFRYPygPo`
- Commit: `ea7e8704502aaa0485aa9dba921416c66c1542b6` plus CLI-only config
  overrides; later branch commits are handoff-only.
- Output root:
  `/root/data/tdm_pilot_sde_200_interval1_35888898_dataset`
- Corrected command summary:
  - `generator_update_interval=1`
  - `max_train_steps=200`
  - dataset:
    `/root/data/.cache/datasets--wlsaidhi--crush-smol_processed_t2v/snapshots/67dd07f2163ad2b3397f8b3d8125b67ca452dc85/combined_parquet_dataset`
  - checkpoints every `100` steps
  - JSONL tracker only
  - validation every `100` steps
  - validation offloads training state and unloads pipeline after validation
- Status observed in logs:
  - Corrected run reached staged parquet dataset, initialized JSONL tracker,
    instantiated validation and EMA, completed step-0 validation, entered
    training, and logged both `grad_norm/student` and `grad_norm/critic`
    starting at step `1`.
  - Through step `100`, every step logged student and critic metrics,
    confirming interval `1` is active and stable through the first half of the
    run. Approximate steady-state step time: `40` seconds.
  - Step `100` started checkpoint save to
    `/root/data/tdm_pilot_sde_200_interval1_35888898_dataset/checkpoint-100`,
    offloaded optimizer state plus teacher/critic for validation, completed
    4-step validation, restored validation state, and resumed training through
    at least step `102`.
  - The local session was interrupted after logs had reached about step `111`.
    After the interruption, `/tmp` worktrees were wiped and had to be
    recreated. Modal volume listing showed only partial committed artifacts:
    `checkpoint-100`, tracker, and validation MP4s for steps `0` and `100`.
    No `checkpoint-200` or step-200 validation files were present, so the
    interval-1 diagnostic is not complete yet.
- Resume plan:
  - Recreate `/tmp/fastvideo-worktrees/issue-775-tdm` at branch
    `issue/775-tdm` and `/tmp/fastvideo-worktrees/interleavethinker-modal` at
    branch `interleavethinker`.
  - Relaunch the same interval-1 command with
    `training.checkpoint.resume_from_checkpoint=/root/data/tdm_pilot_sde_200_interval1_35888898_dataset/checkpoint-100`,
    same output root, and `max_train_steps=200`.
  - After completion, download tracker plus validation MP4s and compare prompt
    0 against teacher/base-DMD/interval-5 artifacts.
- Resume status:
  - Resume app `ap-XVo1dVzlskj0Bs1UUMNbvH` launched from branch
    `issue/775-tdm` using commit
    `ea7e8704502aaa0485aa9dba921416c66c1542b6`.
  - The resumed job loaded
    `/root/data/tdm_pilot_sde_200_interval1_35888898_dataset/checkpoint-100`
    and logged `Checkpoint loaded; resuming from step=100`.
  - It reran step-100 validation after restore, restored optimizer/teacher/
    critic state, then resumed training past the previous interrupted point.
  - As of 2026-07-02 07:47 UTC, logs reached step `150`; each resumed step
    has logged student and critic gradients plus `update_student`, confirming
    `generator_update_interval=1` remains active.
  - The resumed job later advanced normally through step `192`. At
    `2026-07-02 08:15:07 UTC`, the local stream appeared to stop at step `193`
    after `grad_norm/student`; no later streamed `grad_norm/critic`, loss row,
    EMA row, progress-bar advance, checkpoint, validation, or process-exit logs
    followed for several minutes. Later downloaded JSONL showed this was a
    stream/log artifact: the durable tracker contains complete step-`193`
    rows, so the actual stall happened after step `193` completed and before
    step `194` completed.
  - Independent `uvx modal app logs ap-XVo1dVzlskj0Bs1UUMNbvH` checks matched
    the local stream: stdout stopped at the step-193 student-gradient line and
    stderr showed only progress-bar output through step `192`. `uvx modal app
    list` still showed the app as active with one task, so this appears to be
    an in-step hang/deadlock rather than a completed or failed run.
  - No step-200 checkpoint or validation artifacts are expected from this app
    unless the hang unexpectedly resolves before it is stopped. The last
    committed durable artifacts remain `checkpoint-100`, tracker, and
    validation MP4s for steps `0` and `100`.
  - The app was stopped manually with
    `uvx modal app stop ap-XVo1dVzlskj0Bs1UUMNbvH --yes`; the local launcher
    then exited with Modal `RemoteError` caused by "user stopped from CLI".
    After two stop commands, `uvx modal app list` still showed this app in
    `stopping...` state with one task, so Modal may take additional time to
    tear down the deadlocked container.
  - Post-stop Modal volume listing confirmed only these durable artifacts:
    `checkpoint-100`, `tracker`, and the eight validation videos for steps
    `0` and `100`. There is still no `checkpoint-200` or step-200 validation.
  - Persisted artifacts were downloaded to
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_pilot_sde_200_interval1_35888898_dataset/`.
    This local directory contains `tracker/{config.json,metrics.jsonl,
    artifacts.jsonl,files/run.yaml}` plus eight MP4s.
  - Tracker summary:
    - raw JSONL rows: `840`
    - unique steps: `1..193`
    - duplicate metric groups: steps `101..117`, all four row types; this
      matches overlap between the original interrupted job and the resumed
      job.
    - de-duplicated rows: `772`
    - de-duplicated loss/student-grad/critic-grad/EMA rows: `193` each
    - `update_student` true on all de-duplicated steps `1..193`
    - nonfinite scalar metrics: `0`
    - de-duplicated `total_loss`: min `0.1882624775`, max `0.9230759740`,
      mean `0.4580486829`, first `0.3315637112`, last `0.4681196511`
    - de-duplicated `fake_score_loss`: min `0.0004032278`, max
      `0.0016866034`, mean `0.0010733638`, first `0.0015967983`, last
      `0.0014704503`
    - de-duplicated `generator_loss`: min `0.1868882924`, max
      `0.9219567180`, mean `0.4569753191`, first `0.3299669027`, last
      `0.4666492045`
  - Downloaded video sanity check with `ffprobe`: all eight MP4s report
    `832x448`, `77` frames, duration `4.812500`.

What this still will not prove:

- TDM convergence.
- Flow-matching bridge equivalence to the paper in output quality.
- Whether the distilled model is good.
- Whether 4-step Wan TDM beats or matches DMD or Self-Forcing baselines.

## Current Remaining Work

1. Treat the interval-1 hang as bounded but not root-caused.
   The original stopped run was initially misread from the local stream as
   hanging after step-193 student grad and before the critic/loss/EMA rows.
   The downloaded durable JSONL later corrected this: step `193` completed all
   row groups, so the actual stalled interval was after step `193` completed
   and before step `194` completed.
   - 2026-07-02 continuation: code inspection narrowed the exact sequence.
     `grad_norm/*` rows are produced by `GradNormClipCallback` after method
     backward and before optimizer stepping. The JSONL tracker writes and
     flushes the row before logging `JSONL tracker step=...`, so the persisted
     rows are more authoritative than the possibly truncated local stream.
     This motivated the all-rank grad-clip instrumentation; after the durable
     tracker correction, that instrumentation still served to rule out
     step-188..195 grad clipping as the deterministic hang point.
   - Added gated callback instrumentation in
     `fastvideo/train/callbacks/grad_clip.py`: `debug_log` plus
     `debug_log_steps`. With default values off, training behavior is
     unchanged. When enabled, every rank logs begin/end for each role's grad
     clip, local parameter/gradient counts, gradient tensor types/dtypes/
     devices, elapsed time, and returned norm.
   - Local non-test checks for this instrumentation: `python3 -m py_compile
     fastvideo/train/callbacks/grad_clip.py`, `git diff --check`,
     `uvx pre-commit run --files fastvideo/train/callbacks/grad_clip.py`
     (yapf reformatted; mypy hook failed before checking due to the known
     invalid worktree package name `issue-775-tdm`), and direct
     `uvx mypy --explicit-package-bases fastvideo/train/callbacks/grad_clip.py`
     passed.
   - Next Modal diagnostic: run from `checkpoint-100` with
     `generator_update_interval=1`, validation/checkpoint saving disabled,
     and `callbacks.grad_clip.debug_log=true` focused around steps `188..195`.
     If it hangs again, the last debug line should identify whether all ranks
     entered critic clipping and which phase failed to return.
   - First attempted debug Modal app `ap-Km1HANsi5SYrfaretqn4h7` was stopped
     during startup before training because the initial instrumentation used
     the default FastVideo logger behavior, which only logs from local rank 0.
     Patched the debug logger call to pass `local_main_process_only=False` so
     all ranks emit grad-clip begin/end lines. The stop produced the expected
     Modal `RemoteError`/SIGINT logs while the T5 encoder was loading; no
     training-step evidence came from that app.
   - Active all-rank Modal diagnostic: app `ap-og6ZwCuU1gbTgog6T08ub2`,
     commit `3550048bc446d837fc20e948471615720df22c85`, output root
     `/root/data/tdm_debug_interval1_clip_allranks_195_3550048b`.
     It resumes from
     `/root/data/tdm_pilot_sde_200_interval1_35888898_dataset/checkpoint-100`,
     disables validation/checkpoint saves, runs to step 195, and logs
     grad-clip begin/end from all ranks for steps 188-195.
   - Result: the no-validation/no-checkpoint control completed through step
     `195` and committed Modal volume output. Persisted tracker:
     `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_debug_interval1_clip_allranks_195_3550048b/metrics.jsonl`.
     Summary: `380` JSONL rows; steps `101..195` (`95` unique steps);
     `95` student-grad rows, `95` critic-grad rows, `95` loss rows, `95` EMA
     rows; nonfinite scalar metrics `0`. All ranks reported `DTensor` grads
     for both student and critic in the debug window. Critic grad clipping
     returned on every rank at steps `188..195`; step `193` also logged
     critic, loss, and EMA rows. Therefore the prior interval-1 stop is not a
     deterministic grad-clip deadlock in the validation-free training loop.
   - Important correction: the downloaded tracker for the earlier stopped
     interval-1 resume contains complete step-`193` rows (`grad_norm/student`,
     `grad_norm/critic`, loss metrics, and `ema/decay`). The streamed logs had
     appeared to stop after `grad_norm/student`, but the durable JSONL shows
     step `193` completed. The actual stalled interval was after step `193`
     completed and before step `194` completed.
   - Validation-enabled repro: app `ap-4cK2vOsPqcQFNrdO6RbPGy`,
     commit `9fdd20cb7f9de5192eafaa979139aca5bd630b4f`, output root
     `/root/data/tdm_debug_interval1_validation_195_9fdd20cb`. This run keeps
     checkpoint saves disabled but re-enables resume-time validation every
     `100` steps with `offload_training_state=true` and
     `unload_pipeline_after_validation=true`, matching the original
     interval-1 resume's validation behavior. Checkpoint-100 loaded,
     validation offloaded optimizer state plus teacher/critic, generated
     validation videos, restored teacher/critic and optimizer state, restored
     RNG snapshot, then resumed training.
   - Result: the validation-enabled repro completed through step `195` and
     exited normally. Final stream logs showed `Training completed`; the only
     exit warnings were the known non-fatal NCCL
     `destroy_process_group()` warnings. Modal volume commit completed.
   - Persisted tracker:
     `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_debug_interval1_validation_195_9fdd20cb/metrics.jsonl`.
     Summary: `380` JSONL rows; steps `101..195` (`95` unique steps);
     row classes exactly `95` student-grad, `95` critic-grad, `95` loss,
     and `95` EMA rows; missing row groups `0`; nonfinite scalar metrics `0`.
     Scalar ranges: `total_loss` min `0.2086594701`, max `0.9251267314`,
     first `0.3952918947`, last `0.4792018533`; `generator_loss` min
     `0.2075745016`, max `0.9240081310`, first `0.3937318325`, last
     `0.4786063731`; `fake_score_loss` min `0.0004030296`, max
     `0.0016201866`, first `0.0015600767`, last `0.0005954780`;
     `step_time_sec` min `49.8671393080`, max `64.2654010900`, first
     `64.2654010900`, last `49.9735227220`.
   - Debug-window stream evidence for steps `188..195`: all ranks emitted
     begin/end grad-clip logs for both student and critic; all logged gradient
     tensors were `DTensor`, dtype `torch.float32`, on the expected
     `cuda:0..3` devices; student clipping returned in about `0.13..0.14s`
     and critic clipping returned in about `0.03..0.04s`. Step `193` and
     step `194` both completed all tracker rows. Therefore the interval-1
     issue is not reproduced by either the validation-free control or the
     validation-enabled/no-checkpoint control.
2. Decide the next training diagnostic budget. The current evidence supports
   treating the original interval-1 stop as either transient Modal/container
   behavior or a difference in the full original resume command not covered by
   the two controls. The strongest remaining repro check is a full original
   resume-to-200 run from `checkpoint-100` with checkpoint saves enabled and
   validation enabled. If that completes, proceed to the longer interval-1
   convergence pilot; if it hangs, add broader per-step debug logs around
   step `194` beyond grad clipping.
3. If running longer, keep the same prompt 0 comparison workflow:
   teacher 50-step, base Wan DMD 4-step, TDM student 4-step at checkpoints, and
   optional live-student 50-step spot checks. The expected success signal is
   that TDM 4-step starts gaining contrast/detail and prompt-conditioned
   objects relative to the blurred base 4-step DMD baseline.
4. Fix or defer `dcp_to_diffusers` export limitations:
   materialize copied HF symlinks and add a LoRA merge/export path so normal
   inference can load a DCP-exported TDM student.
5. Decide whether the duplicate final `checkpoint-200` save is acceptable as
   existing trainer behavior or should be cleaned up before PR.
6. Before any PR, re-check issue/PR state, rerun Modal tests, and prepare a
   draft PR only if explicitly requested.
