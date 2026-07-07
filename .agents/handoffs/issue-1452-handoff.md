# Issue 1452 Handoff: I2V Dreamverse Support

## Current State

- Issue: #1452, "[Feature] I2V Dreamverse Support"
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1452
- State: OPEN
- Labels: `scope: inference`, `scope: attention`, `scope: model`
- Assignees: none
- Author: Someone45
- Created: 2026-06-11T22:42:23Z
- Updated: 2026-06-12T05:20:25Z
- Repo: hao-ai-lab/FastVideo
- Worktree: `/tmp/fastvideo-worktrees/issue-1452-dreamverse-i2v`
- Branch: `issue/1452-dreamverse-i2v`
- Base: `upstream/main` at e2f4d1a7b (`[feat]: add SwanLab tracker (#1461)`)
- Handoff path: `.agents/handoffs/issue-1452-handoff.md`
- Current stage: Stage 1 deep dive and plan
- Implementation begun: no
- Last updated: 2026-07-07T09:55:02Z

## Authentication And Branch Discovery

- Verified `gh` identity as `macthecadillac` with `gh api user --jq .login`.
- Fetched `origin` and `upstream`; avoided using the `SolitaryThinker` remote.
- `git branch -a --list '*1452*'` found no local or fetched branch containing issue 1452.
- No existing root checkout handoff was found at `.agents/handoffs/issue-1452-handoff.md`.
- Created a new dedicated worktree at `/tmp/fastvideo-worktrees/issue-1452-dreamverse-i2v`.
- `origin/main` was 8 commits behind `upstream/main`; the issue branch was based on `upstream/main` for current upstream behavior.

## Issue Summary

The reporter wants usable image-to-video support in the Dreamverse implementation. They tested by appending `initial_image` to the `session_init_v2` payload. Two behavior problems were reported:

1. Initial I2V outputs were static, with the initial frame remaining still or showing very little motion. The reporter says changing `ltx2_image_crf` to 33 when `image_path` is present and `segment_idx == 1`, while keeping later continuation segments at default CRF 0, improved this.
2. Video degrades as it transitions from clip to clip over 5 second segments. The reporter tried increasing refinement steps from 2 to 3, increasing continuation frames from 9 to 25, decreasing conditioning strength from 1.0 to 0.9, and combinations of these, without success. They suggested testing an initial frame from T2V with the same prompt to determine whether mismatch between prompt and source image is a cause.

## Issue Comments Reviewed

- github-actions requested environment and reproduction details.
- Davids048 confirmed the degradation behavior and suggested it is likely error accumulation in generated video; the issue is less prominent when scenes are more dynamic.
- SolitaryThinker suggested an I2V-specific rewriter LLM system prompt may be needed and asked whether the reporter used gpt oss 120b. Also suggested opening a PR with modifications for iteration.
- CRREO expressed interest in I2V support.

## Related PR State

- `gh pr list -R hao-ai-lab/FastVideo --state open --limit 200 --json ...` was fetched once, but output was too large for a complete manual scan in the terminal output.
- Filtered PR searches:
  - Search `"1452"` returned PR #879, a merged VSA kernel PR whose only match was an image width of 1452. It is not related.
  - Search `"initial_image ltx2_image_crf"` returned no PRs.
  - Search `"Dreamverse I2V"` returned merged/closed historical work:
    - PR #1447 `[infra] Add DGX Spark and multi-architecture CUDA support` (MERGED), which mentions Dreamverse images and LTX2 I2V SSIM but does not address this issue.
    - PR #1540 `[attn] Make FA4 explicit opt-in...` (MERGED), which mentions Dreamverse image FA4 environment behavior but not I2V request quality.
    - PR #1288 `[feat] LTX-2 streaming runtime...` (CLOSED), which landed/covered LTX2 I2V conditioning and continuation infrastructure but is not an active fix for #1452.
  - Active `"i2v"` PRs include #1471 Kandinsky-5 T2V/I2V and #1344 Stable-Video-Infinity Wan I2V. Neither touches Dreamverse.
  - Active `"Dreamverse"` PR #1425 `[feat]: decouple Dreamverse fMP4 streaming from generation` touches `apps/dreamverse/dreamverse/gpu_pool.py`, `session/controller.py`, `tests/test_session_logging.py`, and `worker_ipc.py`; it is about fMP4 media lifecycle and does not address image conditioning. Future implementation should avoid unnecessary edits to those files when possible to reduce conflict risk.
- Filtered issue searches:
  - `"Dreamverse I2V"` and `"initial_image ltx2_image_crf"` returned only issue #1452.
  - Broader `"I2V"` search found related model/support issues such as #697, #868, #973, #813, #1086, #1153, #1219, #785, #711, etc. These are mostly Wan/FastWan/model support or training/distillation issues, not Dreamverse initial-image plumbing.
- No PR draft status has been changed.

## Repo Guidance Read

- `/home/toolbox/.codex/skills/fix-issue/SKILL.md`
- `/home/toolbox/.codex/skills/fix-issue/references/handoff.md`
- `/home/toolbox/.codex/skills/fix-issue/references/stages.md`
- Root `AGENTS.md`
- `apps/dreamverse/AGENTS.md`
- `.agents/lessons/2026-05-22_dreamverse-ci-streaming-imports-need-gpu.md`

Relevant lesson: Dreamverse backend tests may import streaming surfaces that require GPU during collection. Use GPU-backed Modal validation for Dreamverse app tests that touch these imports instead of refactoring import paths just to satisfy CPU-only collection.

## Investigation Notes

- No implementation files have been edited.
- The main checkout `/home/toolbox/FastVideo` is dirty with unrelated untracked files; this issue work is isolated in the worktree above.

### Code Findings

- `apps/dreamverse/arch.md` documents that `initial_image` is validated/persisted by the server, used only for segment 1, and continuation state is kept in the GPU worker for later segments.
- `apps/dreamverse/dreamverse/session/controller.py` currently persists the session image on `session_init_v2` and replaces it on later payloads that carry `initial_image`:
  - `session_init_image = persist_session_init_image(init_data.get("initial_image"))`
  - `replace_session_init_image(payload.get("initial_image"))` appears in reset/simple-generation paths.
  - For generation, `step_image_path = str(session_init_image.file_path) if segment_idx == 1 and session_init_image is not None else None`, then it passes `image_path=step_image_path` into `slot.user_step(...)`.
- `apps/dreamverse/dreamverse/gpu_pool.py` and `worker_ipc.py` carry `image_path` through the worker command payload unchanged.
- `apps/dreamverse/dreamverse/video_generation.py` builds the FastVideo `generate_video` kwargs. It currently hard-codes:
  - `ltx2_image_crf=0.0`
  - `image_path=image_path if segment_idx == 1 else None`
- `fastvideo/pipelines/basic/ltx2/stages/ltx2_image_conditioning.py` confirms the `image_path` path becomes first-frame I2V conditioning:
  - `resolve_ltx2_images` falls back to `[(batch.image_path, 0, 1.0)]`.
  - `DEFAULT_LTX2_IMAGE_CRF = 33.0`.
  - `_preprocess_conditioning_image` H.264 CRF re-encodes the conditioning image to match training quantization; `image_crf <= 0.0` skips that preprocessing.
  - If `image_crf` is not explicitly supplied, it uses `batch.ltx2_image_crf` or the default 33.
- `fastvideo/api/sampling_param.py` and `fastvideo/pipelines/pipeline_batch_info.py` both default `ltx2_image_crf` to 33.0. The SamplingParam comment says the streaming session controller passes `0.0` because it conditions on already-decoded VAE-quality frames. That rationale fits continuation frames better than a freshly uploaded user image.
- Existing tests:
  - `apps/dreamverse/dreamverse/tests/test_session_logging.py` verifies `simple_generate` can pass an uploaded `initial_image` through as an `image_path`, and that a previous non-image segment has `image_path is None`.
  - `apps/dreamverse/dreamverse/tests/test_session_init_image.py` validates image payload parsing/persistence.
  - `fastvideo/tests/api/test_extra_overrides_routing.py` verifies `ltx2_image_crf` is a valid sampling field and reaches `ForwardBatch`.
  - No current test asserts Dreamverse `VideoGenerationWorker.generate_step()` constructs `generate_video` kwargs with a different CRF for uploaded first-frame I2V vs plain T2V/continuation.

### Merits Assessment

- The first reported bug, static or low-motion first I2V output, appears valid and likely caused by Dreamverse overriding the LTX2 default `ltx2_image_crf=33.0` with `0.0` even when a user-uploaded first-frame `image_path` is present.
- The reporter's proposed targeted fix matches current code semantics: use CRF 33 when `image_path` is present for `segment_idx == 1`, while keeping continuation segments at 0 because those use decoded/worker-held continuation frames.
- The second reported bug, progressive "fried" degradation between 5-second clips, also appears plausible and was corroborated by Davids048 as likely error accumulation. Current code already exposes knobs for video conditioning frame count, conditioning strength, video-context noise, and audio conditioning. The attempted changes did not help per the reporter, and current code does not reveal a simple deterministic defect equivalent to the CRF mismatch.
- SolitaryThinker's prompt-rewriter hypothesis is plausible, but current prompt files already contain continuation-specific guidance and no code currently distinguishes user-uploaded first-frame I2V rewrite mode from normal T2V/new-rollout mode. Adding a separate I2V prompt mode would be a broader behavior/design change than needed to fix the confirmed CRF bug.

## Proposed Approaches

1. Minimal targeted CRF fix (recommended)
   - In `VideoGenerationWorker.generate_step`, compute the CRF from whether this request is a true first-segment uploaded-image I2V request: 33.0 when `segment_idx == 1 and image_path` is present, otherwise 0.0.
   - Add a focused unit test around `VideoGenerationWorker.generate_step` using a fake generator and monkeypatched continuation methods to assert:
     - segment 1 with `image_path` passes `ltx2_image_crf=33.0` and the image path;
     - segment 1 without `image_path` keeps `ltx2_image_crf=0.0`;
     - segment 2 keeps `ltx2_image_crf=0.0` and does not pass `image_path`.
   - Low code risk, directly validates the reporter's fix, and avoids PR #1425's session/controller changes.

2. CRF fix plus documentation/comment cleanup
   - Do approach 1.
   - Update the SamplingParam comment or Dreamverse worker comment to clarify that CRF 0 is for decoded continuation frames, while uploaded first-frame I2V should use the LTX2 default CRF.
   - Useful because the current comment could be misread as applying to every Dreamverse image condition.

3. Add an I2V-specific rewrite prompt/mode
   - Add new prompt file/config selection for sessions that start with `initial_image`, then thread that mode through `rewrite_prompt_sequence` or initial rollout generation.
   - This may improve prompt/image alignment, but it touches prompt ownership, devtools config surfaces, frontend/runtime message semantics, and probably tests in files overlapped by PR #1425. It should not be the first fix unless the user explicitly wants broader I2V prompt behavior.

4. Tune continuation quality/degradation knobs
   - Evaluate `LTX2_VIDEO_CONDITIONING_NUM_FRAMES`, `LTX2_VIDEO_CONDITIONING_STRENGTH`, `VIDEO_CONTEXT_NOISE`, refinement steps, and prompt rewrite behavior via qualitative Modal/DGX runs.
   - This is experiment-heavy and less likely to produce a crisp code fix from current evidence. It should be a follow-up after the CRF bug is fixed or after a reproducible quality harness is defined.

## Recommended Plan For Stage 2

Recommended approach: approach 1, optionally with the small comment/doc cleanup from approach 2.

Implementation steps:

1. Re-check issue #1452 comments and open PRs with `gh` before editing.
2. In `apps/dreamverse/dreamverse/video_generation.py`, add a small local helper or inline local variable for the first-segment uploaded-image condition. Avoid new config flags unless requested.
3. Use `ltx2_image_crf=33.0` only for `segment_idx == 1 and image_path` truthy. Keep `image_path=image_path if segment_idx == 1 else None` semantics and keep continuation segments on 0.0.
4. Add focused test coverage in `apps/dreamverse/dreamverse/tests/`, preferably a new worker/request-construction test that stubs `self.generator.generate_video`, continuation save methods, and CUDA sync. Avoid editing `session/controller.py` unless necessary to reduce conflict risk with PR #1425.
5. Optionally adjust the existing SamplingParam/Dreamverse comment to distinguish uploaded first-frame I2V from decoded continuation frames.
6. Do not try to solve the progressive degradation in the same patch beyond recording it as remaining quality risk/follow-up.

Compatibility and performance:

- GPU memory impact should be negligible; this only changes preprocessing of one uploaded conditioning image for segment 1.
- Runtime impact is a small PyAV H.264 encode/decode round-trip for first-frame I2V only. That is already the default LTX2 path and should not affect continuation segments.
- Keeping continuation CRF at 0 avoids perturbing the current decoded-frame continuation path.

## Validation Plan

- Do not run local project tests.
- Stage 2 should use Modal through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`.
- Focused validation:
  - Modal L40S run of the new Dreamverse test(s), likely `pytest apps/dreamverse/dreamverse/tests/test_<new_or_existing>.py -q`.
  - If imports hit the known Dreamverse streaming GPU requirement, keep validation on a GPU-backed Modal job as recorded in `.agents/lessons/2026-05-22_dreamverse-ci-streaming-imports-need-gpu.md`.
  - If feasible, run a small qualitative Dreamverse/LTX2 I2V smoke on Modal/DGX with an uploaded initial image and compare request logs/outputs before and after. This may be expensive and is not needed to prove request construction.
  - Future Stage 4 must run `pre-commit run --all-files` before opening any draft PR.

## Current Recommendation

Proceed with the minimal targeted CRF fix plus focused test coverage. Treat clip-to-clip degradation and I2V-specific prompt rewriting as follow-up investigation unless the user wants a broader Stage 2 scope.

## Validation Status

- No tests or Modal jobs have been run in Stage 1.
- Future implementation validation should use Modal, not local pytest.

## Open Questions

- Whether the first reported static-output issue is still present in current upstream code.
- Whether a minimal first pass should land only the reporter's CRF behavior plus tests, or also introduce an I2V-specific prompt rewrite prompt.
- Whether the clip degradation issue is tractable in this issue or should be documented as a known limitation/follow-up after landing basic I2V support.
