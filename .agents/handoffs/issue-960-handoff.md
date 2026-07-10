# Issue 960 Handoff

## Snapshot

- Issue: #960, "[Feature] Log-linear Sparse Attention"
- URL: https://github.com/hao-ai-lab/FastVideo/issues/960
- State: OPEN
- Labels: help wanted, good first issue, unstale
- Assignees: none
- Author: SingleZombie
- Created: 2025-12-25T08:53:42Z
- Updated: 2026-06-23T05:39:27Z
- Current stage: Stage 1 deep dive and plan complete; awaiting user guidance
- Implementation begun: no

## Workspace

- Repo: hao-ai-lab/FastVideo target, local checkout `/home/toolbox/FastVideo`
- Worktree: `/tmp/fastvideo-worktrees/issue-960-llsa`
- Branch: `issue/960-llsa`
- Branch base: `upstream/main` at `a25313beec11965fce04321b9560b58bcb867504`
- Handoff path: `.agents/handoffs/issue-960-handoff.md`
- `gh` identity verified: `macthecadillac` via `gh api user --jq .login`
- Sandbox notes: `git fetch` and `gh` reads required escalated/out-of-sandbox execution; sandboxed `git fetch` could not write `.git/FETCH_HEAD`, and sandboxed `gh` could not reach `api.github.com`.

## Stage 0 Notes

- Fetched remotes with `git fetch --all --prune`.
- No local or fetched remote branches matching `*960*`.
- No existing `/tmp/fastvideo-worktrees/issue-960-llsa` worktree before creation.
- Created dedicated worktree and branch with:
  - `git worktree add -b issue/960-llsa /tmp/fastvideo-worktrees/issue-960-llsa upstream/main`
- `origin/main` was at `9d909f5f0457ac91f489d5fc8000931f042b72ce`; `upstream/main` was newer at `a25313beec11965fce04321b9560b58bcb867504`. The issue branch was intentionally based on current upstream target code.

## Resume Notes

- 2026-07-10T06:51:06Z: resumed after an interrupted turn. The `/tmp/fastvideo-worktrees/issue-960-llsa` directory had been removed, leaving stale git worktree metadata.
- Recreated the issue worktree with `git worktree add --force /tmp/fastvideo-worktrees/issue-960-llsa issue/960-llsa`; branch restored at pushed handoff commit `cca74ef79`.
- `gh` identity re-verified as `macthecadillac` before GitHub reads.
- Re-read issue #960 with comments. Issue remains open, labels are `help wanted`, `good first issue`, `unstale`, assignees remain empty, and `updatedAt` remains `2026-06-23T05:39:27Z`.
- Re-checked open PRs. Narrow search for `960 LLSA Log-linear Sparse Attention` returned no open PRs.
- Current resume decision: Stage 1 remains complete with no implementation performed. Await user selection between guidance-only/docs, external optional wrapper, or a native kernel port after licensing approval.

## GitHub Context Reviewed

Issue body:

- Reporter presents Log-linear Sparse Attention (LLSA), describes it as an efficient/effective sparse attention method, and asks maintainers for contribution guidance and similar PRs/minimum files.
- Related repository: https://github.com/SingleZombie/LLSA

Issue comments:

- 2025-12-25 SolitaryThinker: pointed reporter to Slack and example attention-backend PRs:
  - VMoba: https://github.com/hao-ai-lab/FastVideo/pull/778
  - SageAttention3: https://github.com/hao-ai-lab/FastVideo/pull/815
  - Noted kernels should be added under the `fastvideo-kernel` package.
- 2025-12-26 SolitaryThinker: linked attention-backend developer docs: https://hao-ai-lab.github.io/FastVideo/attention/developer/
- 2026-02-03 zhisbug: asked whether the reporter was still interested and offered collaboration.
- 2026-02-23 rich7420: asked to take over as a first issue.
- 2026-02-23 SolitaryThinker: approved rich7420 taking it.
- 2026-02-24 rich7420: asked clarifying questions:
  - Whether LLSA's S-Lab License 1.0 non-commercial license is acceptable if optional or external-only.
  - Whether Python + Triton under `fastvideo-kernel/python/` is acceptable for v1.
  - Whether inference-only, no token permutation, is acceptable for v1.
  - Which models/pipelines and tests are expected.
  - What benchmarks should be reported.
- 2026-05-26 github-actions: stale warning.
- 2026-06-22 khizarhayat24: asked whether they can continue work.

Open PR context:

- Queried open PRs with `gh pr list -R hao-ai-lab/FastVideo --state open --limit 200`.
- Narrow search for `960|LLSA|Log-linear|Sparse Attention|SparseAttention|log linear` returned only PR #1183 because its metadata matched generic sparse terminology. It closes #817, not #960, and is an SDPA argument-mismatch fix, so it does not cover this issue.
- No open PR was found that closes or directly implements issue #960.
- No PR draft status was changed.

## Searches And Files Read

- Read root `AGENTS.md` in the issue worktree.
- Read `fastvideo/AGENTS.md` and `fastvideo/attention/AGENTS.md`.
- Read attention developer docs:
  - `docs/attention/developer/index.md`
  - `docs/contributing/attention_backend.md`
  - `docs/attention/index.md`
  - `docs/attention/vsa/index.md`
  - `fastvideo-kernel/README.md`
- Inspected current attention/backend implementation surfaces:
  - `fastvideo/attention/selector.py`
  - `fastvideo/attention/layer.py`
  - `fastvideo/attention/backends/abstract.py`
  - `fastvideo/attention/backends/video_sparse_attn.py`
  - `fastvideo/attention/backends/vmoba.py`
  - `fastvideo/attention/backends/bsa_attn.py`
  - `fastvideo/attention/backends/sla.py`
  - `fastvideo/platforms/interface.py`
  - `fastvideo/platforms/cuda.py`
  - `fastvideo/configs/models/dits/base.py`
  - `fastvideo/pipelines/stages/denoising.py`
  - `fastvideo/pipelines/basic/ltx2/stages/ltx2_denoising.py`
  - `fastvideo/models/dits/wanvideo.py`
  - `fastvideo/models/dits/ltx2.py`
- Searched `.agents/lessons`, `.agents/skills`, and relevant AGENTS paths for LLSA, sparse attention, attention backend, fastvideo-kernel, Triton, and license terms.
- Findings so far:
  - No lesson specific to LLSA or attention-backend porting found.
  - `fastvideo/AGENTS.md` notes attention backend additions belong in `attention/backends/<name>.py` plus selector registration.
  - Some expected AGENTS files such as `fastvideo-kernel/AGENTS.md` and `docs/AGENTS.md` were absent.

Relevant current-code findings:

- `docs/contributing/attention_backend.md` is the current durable guide. It requires backend name/scope, `AttentionBackendEnum`, platform selector wiring, a backend class under `fastvideo/attention/backends/`, call-site support updates, optional `fastvideo-kernel` exports, and tests/docs.
- `fastvideo/platforms/interface.py` currently defines backend enum values including `VIDEO_SPARSE_ATTN`, `BSA_ATTN`, `VMOBA_ATTN`, `SLA_ATTN`, and `SAGE_SLA_ATTN`, but no LLSA value.
- `fastvideo/platforms/cuda.py` performs the concrete string/class resolution and import checks for CUDA backends. New LLSA support would need a CUDA branch.
- Generic `DenoisingStage` allows only `(VIDEO_SPARSE_ATTN, BSA_ATTN, VMOBA_ATTN, FLASH_ATTN, TORCH_SDPA, SAGE_ATTN_THREE)` in its stage-level backend tuple.
- Base DiT config allows many sparse backends globally, but individual model blocks can narrow this. LTX-2 video self attention currently limits itself to `FLASH_ATTN`, `TORCH_SDPA`, and `VIDEO_SPARSE_ATTN`.
- `VideoSparseAttentionBackend` is the closest current pattern for metadata-backed sparse attention: backend class, metadata dataclass/builder, `preprocess_qkv`, `postprocess_output`, and kernel call.
- `VMOBAAttentionBackend` is the closest current pattern for wrapping an external sparse attention package.
- LTX-2 has special VSA metadata handling that only activates for `VIDEO_SPARSE_ATTN` or `SAGE_ATTN_THREE`; LLSA would need separate metadata/permutation handling if it targets LTX-2.

External LLSA repo findings via `gh`:

- Repo: `SingleZombie/LLSA`, default branch `main`.
- Description: official implementation of Log-linear Sparse Attention.
- GitHub license metadata: `Other`.
- `LICENSE.md`: S-Lab License 1.0. Redistribution/use are permitted for non-commercial purposes; commercial use requires contacting contributors. This is not license-compatible with copying code into Apache-2.0 FastVideo without explicit approval or a licensing decision.
- `pyproject.toml`: package name `llsa`, version `0.1`, setuptools build from `src`.
- README says LLSA can replace SDPA with:
  - `llsa_l1_varlen(q, k, v, block_size=16)` for sequence length `< 16384`
  - `llsa_l2_varlen(q, k, v, block_size=16)` for sequence length `>= 16384`
- README constraints:
  - current implementation supports only non-causal attention;
  - token length and `topk` must be powers of two;
  - non-sequential data such as images/videos should be reordered so similar tokens are adjacent.
- Source surface includes Triton kernels and torch-op wrappers under `src/llsa/kernel/`.
- `llsa_l1_varlen`/`llsa_l2_varlen` wrappers use BHSC-shaped tensors (`B, H, S, C`) and compute mean-pooled sparse indices internally.
- External requirements include `accelerate`, `diffusers`, `datasets`, `line_profiler`, `torchmetrics[image]`, `omegaconf`, `lpips`, `einops`, `tensorboard`, and `timm`; not all of these are reasonable as mandatory FastVideo runtime dependencies.

Example PR findings:

- PR #778 (VMoBA) merged, not draft. It added a backend implementation, kernel packaging, config/sample args, denoising metadata support, CUDA/platform enum wiring, tests, and scripts.
- PR #815 (SageAttention3) merged, not draft. It added a backend file, enum/platform wiring, config support, env handling, denoising support, and docs.
- These confirm the current minimal backend contribution pattern but do not resolve LLSA's license constraints.

## Current Hypothesis

- The issue is a contribution-guidance feature request, not a concrete user-facing bug.
- The feature request is valid and still open: no duplicate issue or PR was found beyond #960.
- A direct native port that copies LLSA kernels into FastVideo should not begin until maintainers explicitly approve the non-commercial S-Lab License 1.0 compatibility path or obtain a compatible license grant.
- A safer Stage 2 scope is either:
  - documentation/issue-response guidance only, using the existing backend guide and recording LLSA-specific constraints; or
  - an external optional wrapper backend that imports `llsa` if installed and does not vendor/copy its source, gated behind a new `LLSA_ATTN` backend and strict validation.
- The wrapper approach still needs careful scope because LLSA currently expects BHSC, non-causal self-attention, power-of-two top-k/token constraints, and likely token permutation for good video behavior.

## Possible Approaches

### Approach A: Guidance-only response/docs update

- Files likely touched: maybe `docs/contributing/attention_backend.md` or a new short `docs/attention/llsa/` page if maintainers want durable guidance; otherwise no repo code and only an issue comment in a future GitHub stage.
- Behavior change: none.
- Pros: avoids license risk; directly answers the reporter's minimum-file questions; appropriate for `good first issue`.
- Cons: does not implement LLSA.
- Validation: docs build/pre-commit only if docs changed; no Modal GPU job needed.

### Approach B: External optional `LLSA_ATTN` wrapper, no vendored code

- Files likely touched:
  - `fastvideo/platforms/interface.py`
  - `fastvideo/platforms/cuda.py`
  - `fastvideo/attention/backends/llsa.py`
  - selected supported-backend tuples, probably starting with generic video self-attention only
  - `docs/attention/llsa/index.md`, `docs/attention/index.md`, `mkdocs.yml`
  - focused tests under `fastvideo/tests/attention/` or `tests/local_tests/`
- Behavior change: users who separately install `llsa` and set `FASTVIDEO_ATTENTION_BACKEND=LLSA_ATTN` could route compatible self-attention through LLSA.
- Pros: avoids copying non-commercial code into FastVideo; produces a real integration point; matches VMoBA/SageAttention patterns.
- Cons: still creates a dependency surface around a non-commercial external package; must clearly fail when `llsa` is absent or inputs violate constraints; likely needs per-model opt-in rather than broad default support.
- Validation: source tests for selector/import/shape gating; Modal L40S smoke comparing LLSA output shape/parity against SDPA on a tiny compatible case if `llsa` can be installed in the Modal image; no generation-quality claim without deeper benchmarks.

### Approach C: Native FastVideo-kernel LLSA port

- Files likely touched:
  - `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/`
  - `fastvideo-kernel/python/fastvideo_kernel/__init__.py` and/or `ops.py`
  - possibly `fastvideo-kernel/csrc/attention/` if porting compiled CUDA later
  - same FastVideo backend/platform/docs/tests files as Approach B
- Behavior change: LLSA becomes an in-tree FastVideo kernel/backend.
- Pros: best long-term UX and CI coverage if legally approved.
- Cons: highest risk and largest scope; not appropriate until license/permission is solved; more likely to need parity, benchmarks, and possibly SSIM/generation-quality validation.
- Validation: kernel correctness tests, backend parity tests, Modal L40S runtime tests, benchmarks, maybe model-level smoke/SSIM depending supported models.

## Recommended Stage 2 Plan

Recommended: Approach A or a small preparatory variant of Approach B only after explicit user/maintainer approval of the licensing posture. Do not start a native kernel port from copied LLSA source in this branch without a license decision.

If the user wants code in Stage 2, recommended concrete scope is:

1. Add `LLSA_ATTN` as an external optional backend only, with no vendored LLSA source.
2. Implement `fastvideo/attention/backends/llsa.py` as a thin wrapper that lazily imports:
   - `llsa.kernel.torch_op.flash_sparse_attention_res_1_varlen.llsa_l1_varlen`
   - `llsa.kernel.torch_op.flash_sparse_attention_res_2_varlen.llsa_l2_varlen`
3. Convert FastVideo `[B, S, H, D]` tensors to LLSA `[B, H, S, D]` and back.
4. Start with non-causal self-attention only and fail clearly for unsupported causal or replicated/cross-attention cases.
5. Choose `l1` vs `l2` by sequence length threshold and expose conservative constants internally first (`block_size=16`, top-k defaults). Do not add broad user-facing knobs until tests prove they are needed.
6. Register the backend in `AttentionBackendEnum` and CUDA platform selection.
7. Add the backend only to narrowly compatible supported-backend tuples after checking the target model path, not every model by default.
8. Add docs that state:
   - LLSA must be installed separately from `SingleZombie/LLSA`;
   - LLSA is under S-Lab License 1.0 and FastVideo does not redistribute it in this external-wrapper approach;
   - current limitations: non-causal, power-of-two constraints, video token permutation not yet implemented unless implemented in the patch.
9. Add focused tests for import failure messaging, selector wiring, and a tiny backend call if `llsa` is available; skip or xfail the runtime call when the optional package is absent.

Validation plan for code approach:

- Do not run tests locally.
- Use Modal L40S through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker` for any runtime/GPU validation.
- Minimal validation:
  - Modal command for focused attention/backend tests.
  - If optional `llsa` install is possible in the Modal job, run a tiny LLSA-vs-SDPA shape/numeric smoke on compatible tensor sizes.
  - If a model path is opted in, run one small inference smoke with `FASTVIDEO_ATTENTION_BACKEND=LLSA_ATTN`.
- Future Stage 4 PR gate: `pre-commit run --all-files` must pass before opening any draft PR.

Open questions:

- Should FastVideo accept only an external optional wrapper, or should maintainers first obtain permission/a compatible license for an in-tree port?
- Which first model/pipeline should support `LLSA_ATTN`? Generic Wan/LTX support may need token permutation and model-specific metadata; a broad opt-in could be misleading.
- Should Stage 2 be code, or should this branch only prepare a maintainers-facing issue comment/docs clarification?

## Stage 1 TODO

- Await user guidance for Stage 2.
- No implementation has been performed.
- If Stage 2 proceeds with code, re-check issue/comments/open PRs before editing.
- Before ending Stage 1, commit and push this handoff-only state.
