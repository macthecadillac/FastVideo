# Issue 1592 Handoff

## Identity And Workspace

- Repository: `hao-ai-lab/FastVideo`
- Issue: `#1592` - `[CI] Investigate Vanilla training references under Modal GPU substitution`
- Issue URL: https://github.com/hao-ai-lab/FastVideo/issues/1592
- Branch: `issue/1592-vanilla-modal-gpu-reference`
- Worktree: `/tmp/fastvideo-worktrees/issue-1592-vanilla-modal-gpu-reference`
- Handoff: `.agents/handoffs/issue-1592-handoff.md`
- Base: branch created from fetched `origin/main`, then fast-forwarded to `upstream/main` at `970409962f358afd529b969a378174c849665837`
- Verified GitHub identity: `macthecadillac` via `gh api user --jq .login`
- Authentication note: repository/GitHub commands require approved out-of-sandbox execution because the local filesystem sandbox fails to mount `devpts`.
- Started: `2026-07-12T08:36:52Z`
- Last update: `2026-07-12T09:32:03Z`
- Current stage: Stage 2 complete - evidence supports no production change
- Implementation begun: no

## Issue Snapshot

- State: open
- Created: `2026-07-12T08:29:26Z`
- Updated: `2026-07-12T08:29:47Z`
- Author: `macthecadillac`
- Assignees: `macthecadillac`
- Labels: `scope: training`, `scope: attention`, `scope: distributed`
- Milestone: none
- Requested investigation: run Vanilla on verified L40S and H200 allocations, compare fresh W&B summaries with checked-in references, inspect recent CI history, then retain, refresh, collapse, or pin based on evidence.

## Stage 0 Discovery

- Fetched `origin` and `upstream`; no local/remote branch, worktree, or handoff contained `1592`.
- No open PR directly referenced or closed issue 1592.
- Created the dedicated branch/worktree above. No implementation files have been changed.

## GitHub Context

- The issue was split from open ready-for-review PR `#1591`, which closes `#1586` and is intentionally VSA-only. Its draft state was not changed. Buildkite 4334 passed both Vanilla and VSA after VSA was pinned to `H100!:2`.
- No open PR directly references/closes `#1592`; broader Vanilla/Modal/reference searches found no overlap.
- The issue has one automated `github-actions` welcome comment with no proposed fix or issue-specific evidence.
- Related issue `#1586` contains a reproduced VSA H200 failure. It does not transfer directly because Vanilla's L40S/H200 metrics are close and no analogous Vanilla failure is reported.
- PR `#933`/commit `424fc2b4a` introduced the Vanilla H200 reference while CI already requested `L40S:4`. The PR was a LoRA/distillation fix, gave no baseline rationale, and its author said CI was "really messed up." Provenance is weak, but the values are not shown to be wrong.
- No GitHub draft state, assignee, label, comment, issue, or PR was modified during Stage 1.

## Code And History Findings

- `fastvideo/tests/modal/pr_test.py:194-205`: Vanilla CI requests `gpu="L40S:4"`, disables FA4, and runs `pytest ./fastvideo/tests/training/Vanilla -srP`.
- `fastvideo/tests/training/Vanilla/test_training_loss.py:18-21,123-155`: two torch workers train for five steps, read W&B summary, select A40/L40S/H200 JSON by actual device, and check two timing plus two correctness fields.
- Thresholds: `15.0` for timing, `0.3` for `grad_norm`, `0.0025` for `train_loss`.
- Checked-in L40S/H200 differences: `avg_step_time=1.2160`, `step_time=1.0863`, `grad_norm=0.05144`, `train_loss=0.00008061`. Every checked-in H200 value passes against L40S at current thresholds, so the H200 file is presently redundant as a gate; fresh runs may differ.
- The older A40 summary is materially different and is outside this issue's requested comparison.
- `fastvideo/tests/contract/test_modal_fa4_policy.py` supplies the no-import AST/text pattern for contract-testing any final allocation policy.
- The required `interleavethinker` launcher supports L40S and H100, not H200. Stage 2 needs a temporary uncommitted H200 runner or a separately justified launcher change.
- Official current Modal docs document H100-to-H200 upgrades and `H100!`, not `L40S!`/`H200!`. The installed client's string pass-through does not prove undocumented forms work server-side.
- Historical H200 support makes L40S substitution plausible, but current docs do not confirm it. Actual allocated devices must be checked.

## Recent CI History

- Queried `gh` status histories for the latest 40 upstream PR heads (July 5-12, 2026).
- Exact Vanilla context: 11 successes and 1 failure; no sampled head had a Vanilla failure followed by success.
- The one failure was PR `#1556`/Buildkite 4058, where Vanilla, LoRA, and VSA all failed. Simultaneous failures suggest shared infrastructure rather than a Vanilla-only reference mismatch, but GitHub metadata lacks logs, so this is an inference.
- Recent passes: Buildkite 4334, 4330, 4314, 4285, 4263, 4259, 4253, 4240, 4196, 4147, and 4142.

## Investigation Log

- `2026-07-12T08:36:52Z`: Initialized Stage 1. No code/docs edits, GitHub mutations, or Modal jobs.
- `2026-07-12T08:44:02Z`: Completed GitHub, code, history, CI-status, Modal docs/client, and launcher inspection. No implementation/test/doc edits and no Modal run.
- `2026-07-12T08:50:13Z`: Pre-push re-check confirmed identity `macthecadillac`, unchanged issue/comment state, and no overlapping open PR.
- `2026-07-12T08:59:27Z`: Signed commit `c1ae41fa356ed26a7284fa5518e40720a172662a` verified with key `C943F92E5C32D887` and pushed to `origin/issue/1592-vanilla-modal-gpu-reference`. The repository-local `9970...` key could not sign through the restricted agent; a temporary GPG home with the local public keyring and working agent-backed `C943...` key restored signing without changing persistent git/GPG configuration.\n- `2026-07-12T09:24:50Z`: Signed evidence commit `93c41f20628b97ddb1a256e7ccdd22ad6924ce8c` verified with key `C943F92E5C32D887` and pushed to `origin/issue/1592-vanilla-modal-gpu-reference`.\n- Searches/reads included issue #1592, related issue #1586, PRs #1591/#933/#1556, all associated comments/reviews, open PRs, recent commit statuses, relevant AGENTS/testing/CI docs, Modal launcher/dispatch/contract/test/reference files, and git history for the Vanilla test/reference.

## Current Hypothesis And Merits

The issue is valid as baseline/allocator hygiene investigation, but current evidence does not establish a Vanilla defect or flake. The H200 reference has weak provenance, yet its values agree with L40S under current thresholds, 11 recent jobs passed, and no retry-cleared Vanilla failure appears in the sample. Paired device-verified measurements should precede production changes.

## Alternatives And Recommendation

### Approach 1: Evidence-first decision tree (recommended)

Run unchanged Vanilla on explicitly requested and verified L40S and H200, recover offline W&B summaries using the temporary adapter proven in #1586, and compare each against both references. Repeat only failing/borderline results. Keep valid references; refresh only a reproducibly invalid baseline; change allocation only if hardware variation materially changes gated metrics.

### Approach 2: L40S-only regression

Keep `L40S:4`, reject non-L40S devices, remove H200 selection/reference, and add contract coverage. This reduces maintenance but does not prevent substitution if it exists; `L40S!` is undocumented.

### Approach 3: Explicit L40S/H200 fallback

After validating both devices, declare an ordered fallback list, retain per-device summaries, refresh only invalid data, and contract-test the pool. This makes flexibility intentional but keeps two baselines and may affect queue time/cost.

### Approach 4: Shared L40S/H200 correctness baseline

If repeat measurements show comfortable cross-device margins, share one correctness reference while reporting device identity. This removes redundancy but risks future hardware-specific drift.

Recommendation: approve Approach 1. Do not commit a GPU-policy/reference change before paired data.

## Planned Stage 2 Validation

1. Re-check issue #1592, comments, and open PRs via `gh`.
2. Use `/tmp/fastvideo-worktrees/interleavethinker-launcher` and its required launcher. Add only a temporary uncommitted H200:2 runner if needed, and remove it afterward.
3. Use a temporary uncommitted W&B offline-summary adapter based on #1586; production online-W&B behavior remains unchanged.
4. Run one L40S and one H200 allocation; record actual device names, Modal IDs, full metrics, and exact diffs against both references. Two GPUs match the worker count; use four-GPU reservations only if allocation size matters.
5. Repeat failing or near-threshold results before declaring a reference invalid.
6. Apply only the evidence-selected minimal production change; add focused allocation contract coverage if needed.
7. Validate changed contracts/tests on Modal, run `git diff --check`, GPG-sign commits, push immediately, then perform Stage 3 review/adjudication.
8. Run `pre-commit run --all-files` before a future draft PR. Open no PR before explicit Stage 4 direction; new PRs must be draft and existing draft status must not change.

Pass criteria: verified device identities, captured summaries, exact comparisons, repeatable support for rebaselining, contract coverage for allocation changes, required Modal validation, and passing pre-commit.

## User-Selected Approach

- `2026-07-12`: The user approved the recommended evidence-first decision tree without changes.
- The Stage 2 re-check at `2026-07-12T09:09:55Z` verified identity `macthecadillac`, issue #1592 remains open with only the automated welcome comment, no open PR references/closes #1592, and ready-for-review PR #1591 remains VSA-only. No GitHub state was modified.
- Stage 2 begins with temporary validation-only H200 launcher support and offline W&B summary extraction. These temporary changes must be removed before selecting or committing any production fix.

## Stage 2 Validation Results

- Temporary validation tooling only:
  - Added an uncommitted `H200:2` runner to the `interleavethinker` launcher.
  - Added an uncommitted offline W&B protobuf summary reader and device/summary prints to the Vanilla test.
  - Forced W&B offline mode only in the temporary patch; production online-W&B behavior was not changed.
- Initial detached-job attempts `ap-u9ONgSePleacz7fyGQfJMl` (L40S) and `ap-7F40gXk5DRq1g3eXLWSdzn` (H200) were canceled before checkout completed because `--no-wait` was used without outer `modal run --detach`. This was a launcher invocation error, not a test/device failure.
- Corrected L40S app `ap-dTd6FnWC85KR6uPFtK0MmE` / call `fc-01KXAST529VZ08JVK3RRFGYWBZ`:
  - Two allocated GPUs both reported `NVIDIA L40S`.
  - Result: `1 passed, 1 warning in 119.90s`.
  - Fresh summary: `avg_step_time=2.934632917800002`, `grad_norm=0.1174667701125145`, `step_time=2.5957786740000017`, `train_loss=0.1382552981376648`.
  - Own L40S-reference diffs: `0.45919419781213255`, `0.0016373544931411743`, `0.3382597604031332`, `0.000019259750843048096`; all pass.
  - H200-reference diffs: `1.6751834678733872`, `0.053081467747688293`, `1.4245883413786657`, `0.0000613480806350708`; all pass.
- Corrected H200 app `ap-ui8b9NJk54ZLD7D5KbIPeK` / call `fc-01KXASV0YXWZPGXK7N07F9PN1R`:
  - Two allocated GPUs both reported `NVIDIA H200`.
  - Result: `1 passed, 1 warning in 135.72s`.
  - Fresh summary: `avg_step_time=2.0122275741999998`, `grad_norm=0.11806415766477585`, `step_time=1.1265543279999974`, `train_loss=0.1382402777671814`.
  - Own H200-reference diffs: `0.7527781242733851`, `0.053678855299949646`, `0.04463600462133854`, `0.00007636845111846924`; all pass.
  - L40S-reference diffs: `0.4632111457878696`, `0.002234742045402527`, `1.1309645855968711`, `0.000004239380359649658`; all pass.
- Fresh L40S-versus-H200 diffs: `avg_step_time=0.9224053436000022`, `grad_norm=0.0005973875522613525`, `step_time=1.4692243460000043`, `train_loss=0.000015020370483398438`; all are comfortably within existing thresholds.
- Neither result was failing or near a threshold, so the approved plan did not require a repeat run.
- The temporary launcher and test changes were reversed exactly via their git diff. `/tmp/fastvideo-worktrees/interleavethinker-launcher` is clean and no temporary validation code is present in the issue branch.

## Evidence-Selected Outcome

- Keep the existing L40S/H200 multi-reference behavior unchanged. Both references remain valid under the regression's thresholds, and each fresh run passes against either reference.
- Do not refresh the H200 baseline from one run: it passes, and rebaselining would add churn without fixing a reproduced failure.
- Do not pin the production lane: recent CI history shows no device-dependent Vanilla flake, the explicit L40S diagnostic allocated L40S, and strict L40S syntax remains undocumented.
- Do not collapse references in this issue: although currently redundant under the broad thresholds, retaining manual device-specific evidence is harmless and avoids an unrelated cleanup.
- Production files changed: none. No code commit, Stage 3 code review, pre-commit run, or draft PR is warranted for a no-change investigation outcome.

## Compatibility, Performance, And Documentation

- No model behavior or GPU-memory change is expected. Allocation changes can affect queue time, reservation count, and cost.
- No user-facing docs change is justified. Update durable CI docs only if final lane behavior makes them inaccurate.

## Validation Status

- Local tests: not run; project rules prohibit local tests.
- Modal: not run; Stage 1 forbids Modal jobs.
- Pre-commit: not run; mandatory before future draft PR creation.
- GitHub status inspection: 11 Vanilla successes, 1 simultaneous multi-lane failure, no retry-cleared Vanilla failure.

## Open Questions

- Does current L40S allocation ever yield H200, or is H200 only manual/historical support?
- Do fresh L40S/H200 runs pass their own and each other's correctness references?
- Does Modal accept undocumented `L40S!`/`H200!` strings? This materially affects strict-pin proposals.

## Next Steps

1. Re-check issue/comment/open-PR state.
2. GPG-sign, commit, and push this Stage 1 handoff.
3. Present Stage 1 and wait for the user's approach selection.
4. If Approach 1 is approved, begin Stage 2 paired Modal diagnostics. No implementation has been performed.

## Draft Issue Comment

Status: approved by the user at `2026-07-12T09:32:03Z` after requiring every `#1591` reference to use the `PR` prefix; posting pending.

```markdown
@SolitaryThinker, following up on the Vanilla concern you raised in PR #1591, I ran the current Vanilla regression on explicitly requested L40S and H200 allocations.

Results:

- 2x L40S (`ap-dTd6FnWC85KR6uPFtK0MmE`): `1 passed, 1 warning in 119.90s`
  - `grad_norm=0.1174668` vs L40S reference `0.1158294` (diff `0.0016374`, threshold `0.3`)
  - `train_loss=0.1382553` vs L40S reference `0.1382360` (diff `0.0000193`, threshold `0.0025`)
- 2x H200 (`ap-ui8b9NJk54ZLD7D5KbIPeK`): `1 passed, 1 warning in 135.72s`
  - `grad_norm=0.1180642` vs H200 reference `0.0643853` (diff `0.0536789`, threshold `0.3`)
  - `train_loss=0.1382403` vs H200 reference `0.1383166` (diff `0.0000764`, threshold `0.0025`)

Both timing fields also passed. Each fresh run passes not only against its own device reference, but also against the other device's reference. The fresh L40S/H200 correctness metrics are especially close (`grad_norm` diff `0.0005974`; `train_loss` diff `0.0000150`).

I also checked recent GitHub status history: 11 sampled Vanilla training jobs passed. The only failure occurred when Vanilla, LoRA, and VSA training all failed together, and I found no same-head Vanilla failure-then-success retry pattern like the VSA case addressed by PR #1591.

Based on this evidence, I do not see a current Vanilla regression or device-dependent flake. I recommend keeping the existing multi-GPU reference selection unchanged and closing #1592 without a code change; refreshing or pinning the lane would add churn without fixing a reproduced failure.

Does this resolve the follow-up concern from PR #1591 from your perspective?
```
