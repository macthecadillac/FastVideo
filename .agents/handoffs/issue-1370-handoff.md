# Issue 1370 Handoff

## Issue And Workspace

- Issue: `#1370`, "[Bug] Is the MoE expert routing in _sample_timesteps intended to be flow-shift-aware?"
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1370
- State: open
- Labels: `installation`, `scope: training`, `scope: inference`, `scope: attention`, `scope: distributed`, `scope: model`
- Assignees: none
- Created: `2026-05-18T22:52:36Z`
- Updated: `2026-05-18T22:52:59Z` (unchanged on final Stage 1 re-check)
- Repository: `/home/sandbox/FastVideo`
- Issue worktree: `/tmp/fastvideo-worktrees/issue-1370-wan-moe-flow-shift-routing`
- Branch: `issue/1370-wan-moe-flow-shift-routing`, based on current `upstream/main` at `970409962`
- Handoff path: `.agents/handoffs/issue-1370-handoff.md`
- Current stage: Stage 1 complete, awaiting user guidance
- Implementation begun: no; Stage 1 is investigation-only
- Investigation started: `2026-07-13T03:20:14Z`
- Last Stage 1 update: `2026-07-13T03:31:06Z`

## Stage 0 Resume Check

- No local branch, remote branch visible through `gh`, worktree, or handoff matching issue 1370 was found.
- `git fetch upstream --prune` succeeded before the branch was created.
- `git fetch origin --prune` failed because `/home/sandbox/.ssh/config` contains the unsupported macOS-only `UseKeychain` option. Authenticated GitHub reads remain available through `gh`.
- Verified `gh` identity: `macthecadillac`.
- Created this dedicated worktree from `upstream/main`; this is a new investigation, not a resume.
- Final re-check found no PR for this issue branch.

## GitHub Context

- The reporter argues that Wan2.2 MoE training selects/rescales experts in unshifted `u` space while inference routes at a boundary in shifted timestep `t` space. With `flow_shift=12.0` and `boundary_ratio=0.875`, they calculate that the training split maps near `t=988` rather than the inference boundary `t=875`.
- The reporter proposes deriving the sampling split and expert probability from the index nearest `boundary_ratio * num_train_timesteps` in the scheduler's already-shifted timestep array.
- The reporter separately asks whether the commented high-noise expert rescaling branch should be restored; they correctly observe that its absence exposes the high-noise expert to the full `u` interval.
- One automated welcome comment exists and proposes no technical fix. No maintainer response is present.
- Searches for `1370`, `_sample_timesteps`, `boundary_ratio`, flow-shift terms, and Wan2.2 MoE training found no open or draft PR that claims to address this issue. No existing PR draft status was changed.
- Searches for `boundary_ratio` found only this open issue; no duplicate issue was identified.
- PR #688, merged ready-for-review on `2025-08-07`, introduced Wan2.2 14B MoE inference. Its denoising patch explicitly calculates `boundary_timestep = boundary_ratio * scheduler.num_train_timesteps` and selects the high-noise transformer when `t >= boundary_timestep`.
- PR #804, merged ready-for-review on `2025-09-15`, corrected boundary configuration/override plumbing while retaining that `t`-space semantic.
- PR #818, merged ready-for-review on `2025-10-02`, added Wan2.2 MoE training. Commit `5449d034e` restored the high-noise `u` restriction; `112495b8a` reverted it 28 minutes later. The PR body/comments and final approval do not explain why.
- PR #1103, merged ready-for-review on `2026-02-15`, deliberately reinstated `FlowMatchEulerDiscreteScheduler()` with default `shift=1.0` in `TrainingPipeline.train()` to fix issue #1102, where Wan2.1 SFT produced noisy validation videos. This scheduler overwrite is present in the exact issue-linked commit `17f07bc3` and current upstream.
- PR #880 extends MoE routing to self-forcing inference; `_sample_timesteps` search hits #885, #1164, and #1227 concern other behavior and do not fix this issue.
- External references checked through `gh`: official Wan2.2 declares T2V `sample_shift=12.0`, `boundary=0.875`; Diffusers inference routes at `t >= boundary_ratio * num_train_timesteps`; VideoX-Fun finds a boundary index in its configured training scheduler, but it trains a selected high or low expert separately and does not establish FastVideo's shared-SFT scheduler semantics.

## Code And Merit Findings

- In-scope guidance read: root `AGENTS.md`, `fastvideo/AGENTS.md`, `fastvideo/training/AGENTS.md`, `fastvideo/tests/AGENTS.md`, `docs/training/finetune.md`, `docs/training/overview.md`, and the inference architecture's MoE loading section. Relevant lesson search found no issue-specific pitfall.
- `WanTrainingPipeline.initialize_pipeline()` constructs a shift-aware `FlowUniPCMultistepScheduler`, but `TrainingPipeline.train()` replaces it with an unshifted Diffusers `FlowMatchEulerDiscreteScheduler` before any `_sample_timesteps` call. Therefore the reporter's shifted-array calculation does not describe actual SFT execution at their pinned commit or current upstream.
- PR #1103 and issue #1102 provide behavioral evidence that unshifted training noise sampling is intentional/required for legacy SFT quality. `flow_shift` remains relevant to validation/inference scheduling, not to the SFT noise-density distribution.
- With the runtime shift-1 scheduler and uniform weighting, `boundary_ratio=0.875` corresponds exactly to split index 125: high noise is indices `[0, 125)` / `t >= 875`, low noise is `[125, 1000)` / `t < 875`. Selecting low noise with probability 0.875 is consistent with uniform `t`-space training coverage, even though shifted inference spends a different fraction of its finite steps above the threshold.
- The low-noise branch is correct under those semantics: `u = (1 - boundary_ratio) + u * boundary_ratio` restricts `transformer_2` to the low-noise interval.
- The commented high-noise branch is a real defect: when the high-noise expert is selected, `u` is left in `[0, 1)`, so the high-noise transformer receives mostly low-noise samples. Restoring `u *= 1 - boundary_ratio` makes its training region match inference and is backward-compatible for single-transformer models because it is guarded by `transformer_2 is not None`.
- The issue's shifted-index proposal is not recommended. At shift 12 it would allocate about 63.1% of training updates to the high expert and 36.9% to the low expert, changing the training noise density to mirror inference step spacing and undoing the scheduler correction from #1103. The reporter's wording also leaves the Bernoulli orientation ambiguous: `transformer_2` would need probability `1 - split`, not `split`.
- No focused tests currently cover `_sample_timesteps`, MoE interval membership, expert probability, or the commented high-noise behavior.
- Adjacent risk: `_sample_timesteps` reselects the expert inside every gradient-accumulation microstep, but `train_one_step` steps only the expert selected by the final microstep. If accumulation is greater than one and routing changes, earlier gradients for the other expert are discarded at the next zeroing. This is real but separable from the reported boundary issue and should be an explicitly chosen scope expansion.
- Existing nonuniform `logit_normal` and `mode` weighting behavior affinely maps the sampled density into the selected interval and uses `boundary_ratio` as the expert probability. The recommended minimal fix preserves this behavior; exact conditional-density routing would be a larger semantic change.

## Possible Approaches

### A. Targeted Boundary Fix (Recommended)

- Touch `fastvideo/training/training_pipeline.py`, add `fastvideo/tests/training/test_moe_timestep_sampling.py`, and optionally add a short clarification to `docs/training/finetune.md`.
- Restore the guarded high-noise transform `u *= 1 - boundary_ratio`; retain the unshifted SFT scheduler and existing low-noise probability/interval.
- Add a concise comment explaining that legacy SFT samples unshifted training timesteps while pipeline `flow_shift` controls inference/validation spacing.
- Tests force high and low routing and assert exact boundary membership; also assert single-transformer sampling remains unrestricted.
- Tradeoff: smallest behavior correction and directly answers the valid part of the issue. It intentionally does not fix gradient accumulation routing or redefine nonuniform weighting.
- Risk: existing Wan2.2 fine-tunes that inadvertently relied on high-expert low-noise exposure will change numerically, but that exposure contradicts inference ownership.

### B. Boundary Fix Plus Stable Accumulation Routing

- Apply Approach A and refactor expert selection into a once-per-optimizer-step operation, keeping the chosen expert fixed for all accumulation microsteps.
- Add tests proving only the chosen expert receives gradients/optimizer steps across accumulation.
- Tradeoff: fixes a second concrete MoE correctness bug and makes optimizer behavior coherent. It broadens scope and carries more distributed/training-loop risk than the issue requires.
- Risk: update frequency and RNG consumption change whenever `gradient_accumulation_steps > 1`; GPU training validation becomes more important.

### C. Flow-Shift-Aware Split From The Inference Schedule (Not Recommended)

- Implement the reporter's closest-index idea and derive routing intervals/probabilities from the shifted scheduler.
- Tradeoff: matches the fraction of shifted inference steps assigned to each expert, and resembles VideoX-Fun's separate-expert script.
- Risk: conflates inference step spacing with the SFT training noise distribution, conflicts with the #1103 SFT regression fix, changes T2V high/low update shares from 12.5/87.5 to roughly 63.1/36.9, and needs a policy for nonuniform weighting. This is not supported by current FastVideo behavior.

## Recommended Implementation Plan

1. In `TrainingPipeline._sample_timesteps`, remove the stale commented block and restore the high-noise restriction only when `transformer_2` and a boundary are present. Preserve the current low-noise transform and Bernoulli orientation.
2. Near the unshifted scheduler initialization, document the separation between SFT timestep density and inference `flow_shift`; add a brief user-facing finetuning note if the user wants documentation in scope.
3. Add focused tests using a lightweight fake pipeline/scheduler and patched routing randomness/distributed broadcast. Cover forced high routing (`t >= boundary_timestep`), forced low routing (`t < boundary_timestep`), boundary endpoints, batch sampling, and unchanged single-transformer full-range behavior.
4. If the user selects Approach B, separate expert selection from timestep drawing and call it once before the accumulation loop; add an accumulation regression proving the selected model and stepped optimizer cannot diverge.
5. Keep inference, configs, model loading, and both training-stack boundaries unchanged. Do not migrate legacy Wan training into `fastvideo/train/`.
6. Validate on Modal L40S through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`; do not run project tests locally. Run the focused pytest file and, for Approach B, an appropriate lightweight training-loop contract. No full 14B load should be needed for Approach A.
7. Run pre-commit through the repository hooks for changed paths, then `pre-commit run --all-files` as the mandatory readiness gate. Remember `fastvideo/tests/` is deliberately excluded from pre-commit and should not be linted separately.
8. Commit with GPG signing and push, then run the required Stage 3 review-code/adjudication loop. Do not open a PR until the user explicitly requests Stage 4; any new PR must be draft and existing draft status must never be changed.

## Validation Plan And Pass Criteria

- Modal focused test: all new MoE sampling tests pass on the exact branch patch.
- High-expert pass criterion: every forced high route returns a scheduler timestep satisfying inference's inclusive `t >= boundary_timestep` rule.
- Low-expert pass criterion: every forced low route returns `t < boundary_timestep`.
- Compatibility pass criterion: with no `transformer_2`, sampled indices/timesteps remain the original unrestricted values; shift-1 Wan2.2 expert probabilities remain 0.125 high and 0.875 low for uniform weighting.
- Approach B only: one route decision per optimizer step, all accumulation microsteps use that expert, and only its optimizer/scheduler steps.
- Static/style gate: relevant `pre-commit run --files ...` passes where hooks apply, followed by mandatory `pre-commit run --all-files` before Stage 3 readiness and again before Stage 4 PR creation.
- GPU memory/performance expectation: Approach A adds no tensors or model forwards and should have no material memory or throughput impact. Approach B also adds no model forward but changes route scheduling and needs regression validation.
- SSIM is not required for Approach A because inference code and weights are unchanged. A full 14B fine-tuning quality run is residual risk and likely impractical for this narrow correction.

## Open Questions

- Scope decision: choose Approach A, or explicitly include the gradient-accumulation correction from Approach B.
- Documentation decision is non-blocking: code comment only, or also a short `docs/training/finetune.md` clarification.

## Investigation State

- Searches run: branch/handoff discovery; issue/comment read and final re-check; open PR scan; issue/PR searches for issue number and key symbols; detailed reads/diffs for PRs #688, #804, #818, #880, and #1103; exact issue-linked commit inspection; current symbol/history/blame searches; official Wan2.2, Diffusers, and VideoX-Fun reference reads.
- Files inspected include `fastvideo/training/training_pipeline.py`, Wan T2V/I2V training pipelines, flow-match schedulers, training utilities, Wan configs, inference denoising, training tests, and relevant guidance/docs.
- User-selected approach: none; Stage 2 must not start until the Stage 1 report receives user guidance.
- Files changed: this handoff only.
- Validation: no project tests or Modal jobs run in Stage 1. A standalone arithmetic check confirmed shift-1 high/low proportions 0.125/0.875 and showed the proposed shift-12 proportions 0.631/0.369.
- Handoff commit/push: signed commit `cc50e13db` was pushed successfully to `macthecadillac/FastVideo` branch `issue/1370-wan-moe-flow-shift-routing` over HTTPS using the verified `gh` credential helper. The configured SSH origin remains unusable because of the unsupported `UseKeychain` option.
- Next step: present the Stage 1 report and wait for the user's approach/scope decision.

## Future Gates

- Do not run local project tests; use the prescribed Modal L40S path for Stage 2 validation.
- Run `pre-commit run --all-files` before presenting Stage 3 readiness and again before any Stage 4 draft PR creation.
- Create only a draft PR if the user explicitly requests Stage 4; never change an existing PR's draft status.
