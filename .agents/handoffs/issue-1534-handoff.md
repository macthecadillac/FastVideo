# Issue 1534 Handoff

## Metadata
- Issue: #1534
- Repository: `hao-ai-lab/FastVideo`
- URL: pending GitHub read
- Title: pending GitHub read
- State: pending GitHub read
- Labels: pending GitHub read
- Assignees: pending GitHub read
- Branch: `issue/1534-fix`
- Worktree: `/tmp/fastvideo-worktrees/issue-1534-fix`
- Handoff path: `.agents/handoffs/issue-1534-handoff.md`
- Current stage: Stage 1 - Deep Dive And Plan
- Implementation begun: no
- Created: 2026-07-06T03:43:05Z
- Last updated: 2026-07-06T03:47:18Z

## Stage 0 Resume Or Start
- User selected Issue #1534.
- Checked local refs and existing worktrees for branches containing `1534`; none found.
- Fetched all refs from configured remotes.
- Checked local and remote refs for `1534`; no matching branch found.
- Checked for `issue-1534-handoff.md` in the main checkout and `/tmp/fastvideo-worktrees`; none found.
- Created new dedicated worktree `/tmp/fastvideo-worktrees/issue-1534-fix` from `upstream/main` at `9d909f5f0`.

## Stage 1 Status
- Stage 1 is analysis and planning only. No code, docs, GitHub state, labels, assignments, PRs, or Modal jobs have been changed.
- `gh api user --jq .login` verified `macthecadillac`.
- Next: read project guidance, search current code, evaluate approaches, and report to user before any implementation.

## GitHub Context
- Issue: #1534, `[ci] Promoted Baselines And Reseeding`
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1534
- State: OPEN
- Labels: `scope: docs`
- Assignees: none
- Author: Satyam-53
- Created: 2026-07-02T22:38:41Z
- Updated: 2026-07-05T22:54:27Z
- Body request: extend the performance baseline reseed workflow so maintainers can explicitly promote reviewed scheduled-main records as baselines for a comparable identity.
- Stated scope: promoted baseline metadata; baseline eligibility only for reviewed scheduled-main records; keep failed/calibration/mismatch records for audit; compare against rolling and promoted baselines; reseed/promotion docs for new hardware/software cohorts; do not auto-seed `CALIBRATION_NEEDED` cohorts in Phase 1.
- Acceptance criteria: maintainers can promote reviewed records for a comparable identity; comparator can compare against rolling and promoted baselines; failed records excluded from baseline medians; calibration records do not automatically become baselines; reseed workflow documents v2 identity requirements; tests cover promoted baseline selection and excluded records.
- Relevant files named by issue: `.agents/skills/reseed-performance-baseline/SKILL.md`, `fastvideo/tests/performance/compare_baseline.py`, `fastvideo/tests/performance/hf_store.py`, `docs/contributing/performance_benchmarks.md`.
- Parent: RFC #1374.
- Issue comment reviewed: SolitaryThinker wrote on 2026-07-05 that the owner-approved plan is to park this phase until v2 cohorts have a few weeks of real history; promotion rules should be designed against actual cohort data. The trivial reseed-skill doc fix already landed via #1553. This is a strong current signal against implementing full #1534 immediately without renewed user/maintainer direction.

## Related PRs
- Open PR list checked with `gh pr list`; draft status treated as read-only and not changed.
- Direct PR search for `1534`: no results.
- Direct PR search for `"promoted baseline"`: no results.
- Related issue search for `"promoted baseline"` found #1534, parent epic #1527, docs/migration issue #1536, and RFC #1374.
- #1553 `[misc]: update reseed-performance-baseline skill for the hf_store move (#1545 follow-up)` is MERGED, ready-for-review status before merge, and changed only `.agents/skills/reseed-performance-baseline/SKILL.md` to update stale `hf_store` imports and removed env-var references. It does not implement promoted baselines.
- #1546 `[ci]: add performance fingerprint cohorts` is OPEN, ready-for-review, closes #1530. It adds comparable identity fields, recipe/software/hardware fingerprints, dashboard grouping, and docs. It is part of the prerequisite v2 stack but does not close #1534.
- #1551 `[ci]: emit v2 performance result schema` is OPEN, ready-for-review, closes #1531 and depends on #1546. It emits v2 records and preserves identity/cohort fields. It is prerequisite context for #1534.
- #1560 `[ci] Add exact identity performance statuses` is OPEN, ready-for-review, closes #1532. It adds comparator statuses and a reviewed v2 baseline seeding path for scheduled-main full-suite calibration artifacts, plus docs/reseed guidance. It overlaps the early seeding part of #1534 but intentionally closes #1532, not promoted baseline support.

## Code Investigation
- Read root `AGENTS.md`, `fastvideo/AGENTS.md`, and `fastvideo/tests/AGENTS.md`. `fastvideo/tests/` is pre-commit-excluded; match local style manually if later editing tests.
- Searched `.agents/lessons` and `.agents/skills` for performance/reseed/baseline/promoted pitfalls. No relevant `.agents/lessons` entries were found; the existing `reseed-performance-baseline` skill is the relevant local workflow.
- Current `upstream/main` code behavior:
  - `fastvideo/tests/performance/compare_baseline.py` still compares current records against `load_records_for_model(..., last_n=5, successful_only=True, baseline_eligible_only=True)` keyed by `(model_id, gpu_type)`.
  - `_is_baseline_eligible(run_source, success)` returns true only for successful `scheduled_main` records.
  - Missing baseline on `main` prints `No baseline ... Initializing...`, sets `success=True`, and later marks the record baseline-eligible only if the run source is successful scheduled-main.
  - `fastvideo/performance/hf_store.py:is_baseline_eligible_record()` treats explicit `baseline_eligible=True` as eligible and preserves legacy records missing both `baseline_eligible` and `run_source`; there is no promoted-baseline selector or metadata channel on `main`.
  - `fastvideo/tests/performance/test_inference_performance.py` validates v2 config fields (`workload_id`, `variant_id`, `benchmark_version`) and copies them into raw records, but current `main` does not emit `recipe_fingerprint`, `hardware_profile_id`, or `software_profile_id`.
  - `.buildkite/performance-benchmarks/tests/wan-t2v-1.3b.json` has v2 config identity fields but remains on the older taxonomy (`workload_id=wan-t2v-1.3b`, `variant_id=canonical`, `benchmark_version=1`) until the open v2 stack lands.
  - `docs/contributing/performance_benchmarks.md` states that recipe fingerprinting, hardware/software profile IDs, exact-identity comparison, metric-specific threshold policy behavior, promoted baselines, and dashboard regrouping are separate follow-up changes; rolling comparison remains keyed by `(model_id, gpu_type)`.
- Inspected the open `origin/issue/1532-exact-identity-statuses` branch from PR #1560:
  - Adds comparator statuses (`PASS`, `REGRESSION`, `CALIBRATION_NEEDED`, `RECIPE_MISMATCH`, `INFRA_ERROR`) and exact v2 identity filtering.
  - Adds `fastvideo/tests/performance/seed_baseline.py`, which validates successful scheduled-main full-suite `CALIBRATION_NEEDED` artifacts and writes `baseline_seed=true`, `baseline_eligible=true`, `comparison_status=PASS` seed records.
  - This is not promoted-baseline support. Searches on the branch found only `baseline_seed` / `baseline_reseed` fields; docs still say promoted baselines remain a separate follow-up.

## Merit Assessment And Hypotheses
- The request in #1534 is valid at the design level and matches RFC #1374: maintainers eventually need explicit promoted baselines in addition to rolling medians.
- The issue is not currently a bug causing broken CI on `main`. It is a future workflow/feature ticket with `scope: docs` label but code/test acceptance criteria.
- The current issue comment is decisive context: SolitaryThinker recorded an owner-approved plan to park the promoted-baseline phase until v2 cohorts have a few weeks of real history. This means full implementation now would be speculative and would likely design policy before real cohort data exists.
- The open v2 PR stack (#1546, #1551, #1560) is prerequisite context. Implementing #1534 directly from `upstream/main` would either duplicate that stack or produce a feature against the soon-to-be-obsolete legacy comparator path.
- #1560 already covers one edge that superficially overlaps #1534: first v2 baseline seeding from reviewed `CALIBRATION_NEEDED` scheduled-main artifacts. That still does not satisfy #1534's promoted-baseline comparison requirement.

## Possible Approaches
- Approach A - Defer / park per owner-approved comment (recommended now):
  - Do no implementation in Stage 2.
  - Keep the handoff documenting the investigation.
  - Wait until #1546/#1551/#1560 land and v2 scheduled-main cohort data accumulates, then design promoted-baseline rules from real records.
  - Lowest risk; aligns with maintainer direction; no validation needed beyond this investigation.
- Approach B - Documentation-only clarification:
  - Update docs or the issue only to make the parked state clearer.
  - Not recommended unless maintainers want committed docs; current docs already say promoted baselines remain a follow-up, and issue comments are the right place for PR/branch timing.
- Approach C - Implement after prerequisites land:
  - Start from the post-#1560 exact-identity comparator.
  - Add promoted baseline metadata fields, e.g. explicit promotion marker, promotion reason/operator/source, target exact identity, and timestamp.
  - Extend HF loading/comparator logic to compute both rolling and promoted baselines for an exact identity and fail/report regressions against either.
  - Update reseed/promotion workflow docs and tests for promoted-baseline selection, excluded failed/calibration/mismatch records, and audit-retention behavior.
  - This is the likely eventual path, but should wait for real cohort history.
- Approach D - Implement now on top of the open #1560 branch:
  - Technically possible but not recommended. It would stack more policy on an unmerged branch and conflict with the explicit parked plan.

## Recommended Plan
- Recommend Approach A for Stage 1: stop before implementation and ask the user whether to keep the issue parked or explicitly override that plan.
- If the user explicitly directs implementation despite the parked plan, first confirm whether to base work on `upstream/main` or on the open `issue/1532-exact-identity-statuses` branch after it lands/rebases. Starting from current `main` is likely the wrong base for a durable promoted-baseline implementation.
- If later unparked after prerequisites land, expected implementation sequence:
  1. Re-check issue #1534, comments, open PRs, and merged status of #1546/#1551/#1560.
  2. Rebase/create the #1534 branch on the post-v2 comparator code.
  3. Define a minimal promoted-baseline record marker and provenance fields with direct comparator callers.
  4. Extend `hf_store` helpers only as needed to load promoted records by exact identity.
  5. Extend `compare_baseline.py` to compare current records against rolling and promoted baselines and emit summary/status details for both.
  6. Extend the reseed/promotion workflow docs to explain how reviewed scheduled-main records are promoted and how failed/calibration/mismatch records remain audit-only.
  7. Add focused tests in `fastvideo/tests/performance/test_compare_baseline_policy.py` for promoted-baseline selection and exclusions.
  8. Run Modal L40S targeted tests for performance comparator policy, then pre-commit before any draft PR stage.

## Validation Plan
- Stage 1 performed no runtime validation and no Modal jobs, by rule.
- If implementation is later approved:
  - Run targeted performance policy tests on Modal L40S through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`, not locally.
  - Likely target after prerequisites: `pytest fastvideo/tests/performance/test_compare_baseline_policy.py fastvideo/tests/performance/test_inference_performance_identity.py -q`.
  - Before any draft PR creation, run `pre-commit run --all-files` from the issue worktree and fix all issues.

## Open Questions
- Does the user want to respect the owner-approved parked state, or intentionally override it and begin implementation now?
- If overriding, should the implementation wait for #1546/#1551/#1560 to merge, or stack on the current `issue/1532-exact-identity-statuses` branch?
- What exact policy should promoted-baseline comparison use after real v2 cohort data exists: single promoted record, median of promoted records, latest promoted record, or promoted baseline set with explicit versioning?

## Next Steps
- Commit and push this Stage 1 handoff-only investigation.
- Report Stage 1 findings to the user and ask for the next decision.

## Running Log
- 2026-07-06T03:43:05Z: Initialized handoff in the issue worktree before GitHub and code investigation.
- 2026-07-06T03:45:10Z: Recorded GitHub issue/comment/PR findings. No implementation performed.
- 2026-07-06T03:47:18Z: Recorded code findings, approach analysis, recommended parked/defer plan, and validation plan. No implementation performed.
