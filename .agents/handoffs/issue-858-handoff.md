# Issue 858 Handoff

## Snapshot
- Issue: #858 `[Bug] param.grad is None or gradient is zero when reproducing FastWan`
- URL: https://github.com/hao-ai-lab/FastVideo/issues/858
- State: OPEN
- Labels: stale
- Assignees: jzhang38
- Author: dingangui
- Created: 2025-10-30T03:58:59Z
- Updated: 2026-06-28T05:52:56Z
- Current stage: Stage 1 - Deep Dive And Plan
- Implementation begun: no

## Workspace
- Repo: hao-ai-lab/FastVideo, local source at `/home/toolbox/FastVideo`
- Worktree: `/tmp/fastvideo-worktrees/issue-858-fastwan-vsa-grad-check`
- Branch: `issue/858-fastwan-vsa-grad-check`
- Base: `origin/main` at `9d909f5f0457ac91f489d5fc8000931f042b72ce`
- Upstream checked: `upstream/main` at `afb4f7d3c5730e25facb45f504947615cb14c944`
- Handoff path: `.agents/handoffs/issue-858-handoff.md`

## Stage 0 Resume/Search Notes
- Read `/home/toolbox/.codex/skills/fix-issue/SKILL.md` and required references `references/handoff.md` and `references/stages.md`.
- Read root `AGENTS.md` in the main checkout and issue worktree.
- Ran `git fetch origin` and `git fetch upstream`; fetches completed. `git fetch origin` emitted a known-hosts cross-device-link warning but exited 0.
- Verified GitHub identity with `gh api user --jq .login`: `macthecadillac`.
- Searched branch refs for `858`: no local, origin, or upstream branch matched.
- Checked active checkout for `.agents/handoffs/issue-858-handoff.md`: none.
- Created new worktree and branch with `git worktree add -b issue/858-fastwan-vsa-grad-check /tmp/fastvideo-worktrees/issue-858-fastwan-vsa-grad-check origin/main`.
- `origin/main` is an ancestor of `upstream/main` and 10 commits behind; Stage 1 will inspect current upstream-relevant code as needed and record any base implications.

## GitHub Context Reviewed
- `gh issue view 858 -R hao-ai-lab/FastVideo --json number,title,state,body,labels,assignees,author,comments,createdAt,updatedAt,url,milestone`
- Reporter runs `fastvideo/training/wan_distillation_pipeline.py` for Wan2.1 T2V 1.3B distillation with `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`, 2 GPUs, `--num_frames 77`, latent shape described by collaborator as `20x28x52`, and hits `AssertionError` in `fastvideo/training/distillation_pipeline.py` checking every transformer parameter has a nonzero gradient.
- Comments:
  - `zd-daniel`: "same problem".
  - `BrianChen1129`: FastVideo uses 61 frames for 1.3B training; suspects VSA CUDA kernel gradient bug for 1.3B + 77x448x832 when latent sequence length is not a multiple of 256.
  - `nappengman`: similar issue with variable-shape VSA training; inference works; suspects kernel-level numerical precision; temporary workaround was skipping gradient updates on affected steps.
  - `dingangui`: asks if directly commenting out the gradient check assertions is okay and whether it affects training.
  - `nappengman`: says not to simply skip gradient checking; instead manually empty the optimizer's whole state, e.g. set all to zero, and continue training.
  - `github-actions`: marked stale on 2026-06-28.
- Open PR inventory checked with `gh pr list -R hao-ai-lab/FastVideo --state open --limit 200`.
  - No open PR closes or directly references #858.
  - Targeted searches for `858`, `param.grad`, `AssertionError`, `wan_distillation_pipeline`, `VIDEO_SPARSE_ATTN`, `FastWan`, and `VSA` found no duplicate issue or direct PR fix.
  - Nearby open PRs:
    - #1494 `[feat]: hard-fail on missing/mismatched attention backends; require VSA for FastWan` is ready-for-review and touches attention backend selection/config. It enforces VSA use for FastWan but does not address VSA gradient correctness or zero-gradient handling.
    - #1183 `[bugfix]: fix SDPAImpl.forward() argument mismatch when VSA backend unavailable` is ready-for-review and closes #817; it handles non-VSA backend call signatures, not this training gradient failure.
    - #814 `[self-forcing][7/n] Change Wan DiT to have 0 numerical diff with SF's Wan` is ready-for-review and touches Wan DiT alignment/tests, but does not mention #858 or the training gradient assertion.
    - #1563 fixes FastWan2.2 FullAttn/VSA backend mismatch for issue #864, not this Wan2.1 1.3B VSA training gradient issue.
  - Draft statuses were read only; no PR draft status was changed.

## Investigation Log
- 2026-07-09T06:18:19Z: Stage 1 initialized. No implementation changes made.
- 2026-07-09T06:18Z: GitHub PR inventory and targeted duplicate searches completed. No implementation changes made.
- 2026-07-09T06:30Z: Read `fastvideo/AGENTS.md`, `fastvideo/training/AGENTS.md`, `fastvideo/attention/AGENTS.md`, and `fastvideo/models/AGENTS.md`. Searched `.agents/lessons` for VSA/gradient/distillation terms; no relevant lesson matched.
- 2026-07-09T06:41Z: Inspected current legacy distillation train step, VSA metadata/kernel routing, Wan VSA blocks, training tests, and history around the reported assertion. No implementation changes made.

## Current Hypothesis
- The exact reported crash is stale against current `origin/main`: the `for param in self.transformer.parameters(): assert param.grad is not None and param.grad.abs().sum() > 0` check existed at reporter commit `404314d0` (`fastvideo/training/distillation_pipeline.py:1001-1003`) but is absent on the issue branch and on `upstream/main`.
- The assertion was removed by `424fc2b4a` / PR #933 (`[bugfix] [lora] [distillation] Fix lora distillation bug`) along with equivalent fake-score gradient assertions.
- There were also several post-report VSA fixes that plausibly address the root cause behind missing/zero gradients:
  - `5a549af82` / PR #925 fixed backward kernel block-size computation.
  - `2b8c937ab` / PR #944 added VSA padding logic.
  - `0224ea9a1` / PR #1015 added an autograd-enabled block-sparse attention wrapper in `fastvideo-kernel` and updated FastVideo to call the new kernel wrapper signature.
  - `00ec3e738` / PR #1517 computes VSA top-k from padded block count instead of unpadded token count.
- Current code still has runtime gradient hazards if a specific VSA kernel/configuration silently produces bad gradients, but the current user-visible behavior is no longer the reported assertion. Current gradient handling clips finite norms and steps optimizers; it does not verify every parameter has nonzero gradient.
- The reporter's 77x448x832 / `num_latent_t=20` shape maps to Wan post-patch sequence shape `20x28x52 = 29120`. This is not divisible by 256, matching the collaborator's suspicion, but current default `VSA_TILE_SIZE=(4,4,4)` has tile volume 64, and `29120 / 64 = 455` exactly. Current metadata uses 455 padded KV blocks for top-k.
- Current docs state the 1.3B distilled model was trained on 61x448x832, while the current committed 1.3B DMD+VSA SLURM example uses `num_latent_t=21` / `num_frames=81`. The reporter used 77 frames at the old commit. This supports treating 77-frame training as at least historically questionable, but it is not enough by itself to justify a new runtime guard.

## Alternatives And Plan
- Option A - preferred: validate current main before code changes. Run a Modal GPU repro/smoke on current code for DMD+VSA with the reporter-like shape (`num_latent_t=20`, `num_height=448`, `num_width=832`, `num_frames=77`, `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`) and a very small step count. If it passes without the old assertion or gradient/kernel failure, treat #858 as already fixed by later distillation/VSA work and prepare a no-code status/closure note. If it fails, use the new failure as Stage 2 implementation input.
  - Touches: likely no code unless validation fails; possibly handoff/PR body only.
  - Tradeoff: spends GPU time and may need H100/B200-class memory because the reporter's A100 80GB run was slow and the full shape may not fit on L40S.
- Option B: add a targeted lightweight regression for current metadata/math, not full training. Add a CPU test for the reporter's VSA metadata shape (`raw_latent_shape=(20,56,104)`, `patch_size=(1,2,2)`) to lock `dit_seq_shape=(20,28,52)`, `variable_block_sizes.numel()==455`, and `_compute_cur_topk(..., VSA_sparsity=0.8)==91`.
  - Touches: `fastvideo/tests/attention/test_video_sparse_attention_metadata.py`.
  - Tradeoff: cheap and deterministic, but it does not validate kernel backward or DMD gradients; it would mainly protect the #1517 top-k/padded-block logic for this shape.
- Option C: add or adjust a DMD+VSA training smoke. Extend `fastvideo/tests/training/distill/test_distill_dmd.py` or add a new targeted test that sets `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`.
  - Touches: `fastvideo/tests/training/distill/` or `fastvideo/tests/training/VSA/`.
  - Tradeoff: closest to the issue, but likely expensive and flaky if it uses the full 77x448x832 shape; a reduced-shape smoke would not reproduce the reported trigger.
- Option D: documentation-only clarification. Update distillation docs/examples to clarify supported/recommended frame counts for 1.3B DMD+VSA and mention that older gradient assertions were removed.
  - Touches: `docs/distillation/dmd.md` and/or `examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P/README.md`.
  - Tradeoff: useful if maintainers want to resolve as configuration guidance, but it should not replace validation because the issue reported a runtime failure.

Recommended plan: Option A first. Validate current code against the reporter-like shape on Modal. If it passes, do not patch runtime code; report that #858 is already resolved by post-404314d0 fixes and prepare a concise GitHub response. If it fails, implement the smallest fix dictated by that new failure. Option B can be added after validation if maintainers want a low-cost regression for the exact shape math.

## Validation Status
- No validation run yet. Per Stage 1 rules, no Modal or tests are run before the user approves implementation.
- No local tests were run; FastVideo rules forbid local test execution.
- Suggested Stage 2 validation if user approves Option A:
  - Re-check issue/comments/open PRs with `gh`.
  - Use `fastvideo/tests/modal/launch_l40s_job.py` from branch/worktree `interleavethinker`.
  - Prefer H100 or B200-class GPU if L40S memory is insufficient for reporter-like `num_latent_t=20`, 448x832, DMD+VSA.
  - Command should set `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN` and run a short DMD distillation invocation or a targeted test script using current `wan_distillation_pipeline.py`.
  - If code changes later happen, run focused Modal validation and the Stage 3 `review-code` / adjudicator loop before any draft PR message.
- Mandatory future PR gate: `pre-commit run --all-files` before any draft PR creation; no draft PR should be opened until explicit Stage 4 direction.

## Next Steps
- Commit and push this handoff-only Stage 1 commit.
- Ask the user to choose:
  - validate/no-code-first path (recommended),
  - add lightweight metadata regression,
  - add/adjust DMD+VSA training smoke,
  - documentation-only clarification,
  - or another concrete implementation direction.
- Do not implement code, comment on GitHub, assign/label, run Modal, or open a PR until the user gives Stage 2 guidance.
