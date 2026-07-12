# Issue 1105 Handoff

## Workload State

- Issue: `hao-ai-lab/FastVideo#1105`, "[Bug] full finetune wan-2.1-1.3 model failed"
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1105
- State: open; labels: none; assignees: `alexzms`
- Created: `2026-02-15T05:04:07Z`; updated: `2026-04-26T08:51:50Z`
- Repository: `/home/sandbox/FastVideo`
- Dedicated worktree: `/tmp/fastvideo-worktrees/issue-1105-wan-finetune-quality`
- Branch: `issue/1105-wan-finetune-quality`, created from `origin/main` at `19a51a1fe`
- Handoff: `.agents/handoffs/issue-1105-handoff.md`
- Current stage: Stage 1, deep dive and plan
- Implementation begun: no. Stage 1 is analysis-only; this handoff is the only content write.
- Stage 1 started: `2026-07-12T15:04:46Z`

## Identity And Sandbox

- `gh api user --jq .login` returned `macthecadillac` on `2026-07-12`.
- GitHub reads use only `gh` against `hao-ai-lab/FastVideo`.
- The managed filesystem sandbox failed to initialize with `bwrap: Can't mount devpts on /newroot/dev/pts: Permission denied`; required reads and git/GitHub operations were rerun with approved escalated permissions.

## Stage 0 Discovery

- Fetched `origin` and searched local/remote refs matching `*1105*`: none found.
- Queried `macthecadillac/FastVideo` matching refs under `heads/issue/1105`: none found.
- Checked the active checkout for a tracked `.agents/handoffs/issue-1105-handoff.md`: none found.
- Searched open upstream PRs for `1105`, issue references, and Wan finetune-quality terms: no related open PR found.
- No prior handoff or PR exists, so this is a new investigation.
- The fork's `origin/main` was `19a51a1fe`; fetched `upstream/main` was `970409962`, three commits ahead. The upstream-only changes concern AnyFlow and attention/CI work; they do not change the legacy reproduction script or legacy Wan training loop. Rebase the issue branch onto current upstream before Stage 2 implementation.

## GitHub Snapshot

- Reporter: `liuuzexiang`.
- Report: latest code running `bash examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v.sh` produced poor validation video quality while full-finetuning Wan 2.1 1.3B.
- Environment included PyTorch `2.10.0`, CUDA toolkit `12.6`, and `fastvideo-kernel 0.2.5`.
- Comment by `jzhang38` (`2026-02-26`): suggested overtraining and reported acceptable quality at step 100.
- Comment by `alexzms` (`2026-04-26`): asked whether the issue persists with the new framework under `fastvideo/train` and `examples/train`.
- No commenter supplied a concrete patch. The two current hypotheses are overtraining/configuration behavior in the legacy example and possible obsolescence after migration to the new training framework.
- The issue is 147 days old as of `2026-07-12` and has had no reporter response since creation.
- Final focused open-PR searches for `#1105`, `#1102`, `full finetune Wan`, the issue title, branch names containing `1105`, and the reported symptom returned no related open or draft PR.

## Related Issue And PR Timeline

- Issue #1102, from the same reporter and with the same command/environment, was opened at `2026-02-14T15:49:04Z`. It specifically reported that validation was normal at step 0 but entirely noisy after 200 steps despite a small loss.
- PR #1103, "[bugfix] Fix failed kernel publish and SFT regressions", explicitly fixed #1102 and merged at `2026-02-15T00:19:17Z`. PR state was ready-for-review before merge; no draft state was changed.
- PR #1084 had accidentally removed `self.noise_scheduler = FlowMatchEulerDiscreteScheduler()` from `TrainingPipeline.train()` on `2026-02-10`. That made the legacy Wan trainer reuse its inference `FlowUniPCMultistepScheduler` for timestep/sigma preparation. PR #1103 restored the flow-match training scheduler. Current code retains the fix at `fastvideo/training/training_pipeline.py:536`.
- Issue #1105 was opened by the same reporter only 4 hours 45 minutes after #1103 merged. Its wording changed from "all noisy at step 200" to generally "worse" validation quality. This supports a two-part timeline: the catastrophic scheduler regression was fixed by #1103; aggressive recipe defaults still degrade quality after early steps.
- PR #1159 introduced the maintained modular training stack on `2026-03-09`. Its test plan states that Wan finetuning matched the legacy loss curve. PR #1177 added the current basic modular Wan recipe on `2026-03-26`. Both are merged, not open, and neither closes #1105.

## Investigation Log

- Read root `AGENTS.md`, `fastvideo/training/AGENTS.md`, `fastvideo/train/AGENTS.md`, `fastvideo/tests/AGENTS.md`, the `fix-issue` handoff/stage references, `fastvideo/train/README.md`, `examples/train/README.md`, `examples/train/configs/README.md`, and `docs/training/finetune.md`.
- Searched `.agents/lessons/`; the existing lessons concern model-port dtype/packing and Dreamverse CI, with no applicable Wan finetuning-quality lesson.
- Inspected the legacy reproduction script and history, `WanTrainingPipeline`, `TrainingPipeline` initialization/loss/validation paths, the modular `WanModel`, `FineTuneMethod`, validation callback, current Wan YAML, legacy/modular tests, and the tracked nightly Wan overfit video reference.
- The public legacy script's core optimization defaults have been unchanged since June 2025: 5,000 steps, effective batch 8, learning rate `5e-5`, CFG dropout `0.1`, and weight decay `1e-4`. It first validates after 200 steps.
- The legacy training-loss regression has used the same Crush-Smol processed dataset with learning rate `1e-6`, CFG dropout `0.0`, and weight decay `0.01` since June 2025. Its five-step checks cover loss/gradient behavior, not output quality.
- The maintained modular Wan recipe uses effective batch 8, learning rate `1e-6`, CFG dropout `0.0`, weight decay `0.01`, 4,000 steps, and validation every 100 steps.
- The collaborator's comment that step 100 is acceptable is consistent with the issue #1102/PR #1103 timeline and with the public script's first validation occurring only at step 200.
- The legacy and modular implementations use the same normalized latents, uniform flow-matching timestep sampling, noisy-input interpolation, `noise - clean_latents` target, and MSE loss. Both validate against the live training transformer through the standard Wan inference pipeline. No current loss, validation-weight, or model-mode defect was found.
- The legacy launcher passes `--ema_start_step 0`, but the base finetuning pipeline does not create or validate with an EMA object; this is unused configuration, not evidence that validation is reading stale weights.
- The user-facing `docs/training/finetune.md` still recommends the legacy script and describes `1e-5` to `5e-5` as typical full-finetune learning rates. That guidance conflicts with both the long-standing legacy regression (`1e-6`) and maintained modular recipe (`1e-6`).
- No local tests or Modal jobs were run in Stage 1, as required.

## Current Hypothesis And Merit Assessment

- **Confirmed historical defect:** PR #1084 removed the flow-match training scheduler reset and caused the catastrophic step-200 noisy output reported in duplicate issue #1102. PR #1103 restored it before #1105 was filed, and current code contains the fix.
- **Current valid defect:** the public, documented Crush-Smol recipe remains materially more aggressive than every maintained/tested Wan recipe. On a small demonstration dataset, `5e-5` for up to 5,000 optimizer steps with the first post-training preview only at step 200 makes rapid quality collapse plausible and matches the maintainer's direct observation.
- **Not proved:** the reporter gave no checkpoint step, loss trace, exact commit, or post-#1103 confirmation in #1105. Stage 1 therefore cannot prove one exact hyperparameter is sufficient without a controlled run.
- **Impact:** users following the primary full-finetune guide can spend four GPUs and substantial time producing a degraded checkpoint even though safer settings already exist in CI and the modular stack.
- **Scope:** recipe and durable documentation first. Current training math should not be changed without new evidence.

## Approaches And Recommendation

### Approach A: Correct the shipped legacy Crush-Smol recipe (recommended)

- First run a controlled current-versus-candidate experiment on Modal. Candidate settings should start from the already-tested values: `learning_rate=1e-6`, `training_cfg_rate=0.0`, `weight_decay=0.01`, and validation every 100 steps. Use the experiment to choose a defensible default step limit rather than guessing between 100, 200, or 4,000.
- Update `examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v.sh`, its README, and `docs/training/finetune.md`. Remove or narrow the unsupported generic `1e-5` to `5e-5` Wan guidance and explain checkpoint selection/validation cadence as durable behavior, without issue/PR-specific wording.
- Keep `fastvideo/training/` algorithm code unchanged. This is the smallest fix for the exact documented command and is permitted maintenance of a shipped legacy recipe.
- Risk: quality is dataset-dependent, so the candidate must be validated at more than one checkpoint. Lower learning rate does not increase GPU memory; more frequent validation increases wall time only.

### Approach B: Move the documented Crush-Smol quickstart to `fastvideo/train`

- Make the modular Wan YAML directly runnable with the existing Crush-Smol preprocessing output and validation JSON, and update the quickstart/docs to launch it through `examples/train/run.sh`. Clearly mark the old shell launcher as legacy or retire it if compatibility permits.
- This follows the maintainer comment and current architecture direction, and removes duplicated hyperparameters across the two stacks.
- Risk: broader behavioral change, different HSDP layout/checkpoint format, and migration/documentation work. The repository rules prohibit silently migrating an existing pipeline, so this requires explicit user approval and stronger end-to-end validation.

### Approach C: Treat #1105 as already fixed/stale and make no repository change

- Rely on #1103 for the historical scheduler defect and the unanswered request to try the modular stack.
- Lowest engineering cost, but it leaves the exact public command and docs inconsistent with tested settings. Not recommended.

**Recommendation:** Approach A. It fixes the user-visible reproduction path with the least blast radius, preserves stack boundaries, and can be grounded in a controlled quality comparison before choosing final values.

## Validation Plan

- Before editing in Stage 2, re-check issue #1105, all comments, issue #1102, PR #1103, and the open PR list; then rebase onto current `upstream/main`.
- Use `fastvideo/tests/modal/launch_l40s_job.py` from branch/worktree `interleavethinker` with 4x L40S and a local patch for candidate files. The launcher supports `L40S:4`.
- Download/reuse `wlsaidhi/crush-smol_processed_t2v` and run a controlled 200-step legacy comparison at fixed seed and prompts: current defaults versus the candidate safe optimizer/data defaults, logging at steps 0, 100, and 200.
- Compare loss/grad norm plus validation artifacts for text consistency, temporal coherence, and visible noise/collapse. Use the repository's video-quality evaluation workflow and preserve commands/artifact paths/results in this handoff. If neither candidate clearly improves step 200 while preserving step 100, do not patch based on hyperparameter conjecture.
- After selecting values, run the existing legacy five-step Wan training-loss regression on Modal L40S with the proposed files and require all metric thresholds to pass.
- Add a focused, network-free contract check only if it can verify the documented recipe/default relationship through a structured surface; do not add brittle shell-text parsing solely to satisfy a checklist.
- Run relevant docs/config checks through pre-commit. Before any future draft PR creation, run mandatory `pre-commit run --all-files` and require every hook to pass.
- No targeted Wan T2V inference SSIM is required if only recipe/docs change because inference/model code is untouched. If implementation expands into model or pipeline code, run the targeted Wan T2V SSIM regression on L40S.
- `pre-commit run --all-files` is mandatory before any future draft PR creation.
- A future new PR must be draft-only. Existing PR draft status must never be changed.

## Open Questions And Next Steps

1. User decision: select Approach A, B, or C. No technical question blocks Approach A.
2. If Approach A is approved, begin Stage 2 with the controlled Modal comparison before selecting exact final defaults.
3. If Approach B is approved, confirm whether backward-compatible legacy launcher retention is required before changing the documented entrypoint.

## Stage 1 Completion

- Stage 1 analysis completed on `2026-07-12`.
- No implementation has been performed. Only this required handoff file has been written.
- Rebased the issue branch onto fetched `upstream/main` at `970409962` before persistence.
- Created GPG-signed handoff commit `9fc5afd3212c2266701ecce886fc3e53117ab094`; signature verified as good for `Mac Lee <macthecadillac@gmail.com>`.
- Pushed `issue/1105-wan-finetune-quality` to `origin` (`macthecadillac/FastVideo`) and configured the upstream tracking branch.
- Immediately before that push, `gh` identity was re-verified as `macthecadillac`, issue #1105 and all comments were unchanged, and the focused overlapping-open-PR search remained empty.
- Awaiting user guidance before Stage 2.

## PR State

- No PR exists for this branch or issue.
- No PR or draft PR has been opened, and no existing PR draft status has been changed.
