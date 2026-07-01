# Issue 775 TDM Exploration Handoff

Generated: 2026-06-30

## Worktree

- Branch: `issue/775-tdm`
- Worktree: `/tmp/fastvideo-worktrees/issue-775-tdm`
- File: `.agents/exploration/issue-775-tdm.md`
- GitHub identity checked before reads: `gh auth status --hostname github.com`
  reported active account `macthecadillac`.

## GitHub State Checked

- Issue: https://github.com/hao-ai-lab/FastVideo/issues/775
- Title: `[Feature] TDM`
- Author: `fenght96`
- Created: 2025-09-01
- Updated: 2026-05-31
- State: open
- Labels as of 2026-06-30: `good first issue`, `contribution-needed`, `stale`
- Assignees: none

Issue body asks whether FastVideo has a plan for TDM:
`Learning Few-Step Diffusion Models by Trajectory Distribution Matching`
(`arXiv:2503.06674`).

Comments reviewed:

- 2026-02-04: `zhisbug` said it is interesting to add and the team would
  consider whether someone can work on it.
- 2026-05-31: stale bot marked the issue stale because of 90 days inactivity.

Open PR list was checked with `gh pr list --repo hao-ai-lab/FastVideo
--state open --limit 100`. No open PR title/head branch obviously targets TDM
or issue 775. Narrow searches were also run:

- `gh pr list --repo hao-ai-lab/FastVideo --state all --search "TDM"` -> no PRs.
- `gh pr list --repo hao-ai-lab/FastVideo --state all --search "775"` -> only
  unrelated/adjacent merged PR #755 (`[Feature] Support Lora for DMD`,
  branch `lora_distill`), which may matter if a TDM LoRA training path is
  chosen.

## External TDM Sources Read

- Paper: https://arxiv.org/abs/2503.06674
- Project page: https://tdm-t2x.github.io/
- Official repo, read through `gh`: https://github.com/Luo-Yihong/TDM
- Official training demo: `Luo-Yihong/TDM/train_tdm_demo.py`

High-level paper/repo notes:

- TDM is a data-free few-step distillation method. Training needs prompts, not
  ground-truth images/videos.
- The method matches distributions of generated trajectories rather than only
  final samples. The project page describes this as score divergence over
  student trajectories.
- The project page also highlights TSAM, a sampling-steps-aware matching
  objective so one student can adapt to different NFE counts.
- Official repo is minimal: `README.md`, `requirements.txt`, `assets/`, and
  `train_tdm_demo.py`.
- Official released LoRAs include SD3.5, SD3, Dreamshaper, and CogVideoX-2B.
  The README gives a CogVideoX-2B text-to-video example using 4 NFE and notes
  the generator was trained on timesteps `[999, 856, 665, 399]`.
- The demo script trains a trainable generator and a trainable fake-score model
  against a frozen teacher. It alternates fake-score and generator updates,
  builds a generated/noisy trajectory with `generate_new`, and uses helper
  routines equivalent to `Predictor.predict`, `Predictor.add_noise`, and
  `Predictor.obtain_mixed_noise`.

## FastVideo Codebase Fit

Relevant repo guidance:

- New training work should go under `fastvideo/train/`, not legacy
  `fastvideo/training/`.
- `fastvideo/train/AGENTS.md` says new methods subclass `TrainingMethod` and
  live under `fastvideo/train/methods/<family>/`.
- `fastvideo/training/AGENTS.md` marks the legacy monolithic stack as
  maintenance mode and says new training methods should not be added there.

Closest existing implementation:

- `fastvideo/train/methods/distribution_matching/dmd2.py`
- `fastvideo/train/methods/distribution_matching/self_forcing.py`
- Example configs:
  - `examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml`
  - `examples/train/configs/distribution_matching/wan/self_forcing_causal_t2v.yaml`
- User-facing docs:
  - `docs/training/train_infra.md`

`DMD2Method` already has important building blocks:

- Role models: trainable `student`, frozen `teacher`, trainable `critic`.
- Text-only simulation mode via `rollout_mode: simulate`.
- Separate student and critic optimizers/schedulers.
- Fixed denoising step lists through `method.dmd_denoising_steps`.
- CFG teacher score support via `real_score_guidance_scale`.
- Student rollout that creates synthetic latent trajectories from noise.

Likely TDM integration shape:

- Add a new method in `fastvideo/train/methods/distribution_matching/`, probably
  `tdm.py`.
- Consider subclassing `DMD2Method` only if the shared optimizer/role-model
  handling stays compatible. The loss and trajectory generation are different
  enough that a sibling `TDMMethod(TrainingMethod)` may be clearer.
- Start with Wan 2.1 T2V 1.3B because FastVideo already has `WanModel`,
  DMD2 configs, validation callbacks, and DMD inference pipelines.
- First config should probably be text-only/data-free and LoRA-capable, since
  official released TDM artifacts are LoRAs and training is prompt-only.

## Main Implementation Questions

1. Target model: should first support be Wan 2.1 T2V 1.3B, Wan 2.2, or just
   import official CogVideoX-2B TDM LoRA? FastVideo does not appear to have a
   first-class CogVideoX pipeline, so a Wan training method is the more natural
   native contribution.
2. Scope: method only, or also a shipped pretrained checkpoint/preset? The
   GitHub issue asks "plan", not specifically a port of official weights.
3. LoRA versus full fine-tune: official examples emphasize LoRA. Verify current
   `fastvideo/train/utils/lora.py` and model role config support the desired
   student/critic setup before implementing.
4. Timestep schedule: official CogVideoX example uses `[999, 856, 665, 399]`;
   FastVideo DMD2 examples use `[1000, 750, 500, 250]`. Do not assume the DMD2
   schedule is valid for TDM quality.
5. TSAM: decide whether first PR implements only fixed 4-step TDM or includes
   sampling-steps-aware matching. TSAM expands config and validation surface.
6. Validation: for a method-only PR, use small fake-model/unit tests locally in
   code, then run real validation on Modal L40S. Do not run local tests per repo
   rules.

## Risks And Pitfalls

- The official TDM demo is diffusers/accelerate image-model code with
  DDPM-style alpha/sigma helpers. FastVideo Wan training uses flow matching
  scheduler utilities and `[B, T, C, H, W]` latent conventions. Scheduler math
  must be translated deliberately, not pasted.
- Existing `DMD2Method._critic_flow_matching_loss` trains the critic against
  `noise - generator_pred_x0`; the TDM demo trains fake score on noised
  generated latents with mixed-noise importance weighting. Treat this as a
  distinct objective.
- Preserve `TrainingMethod.cuda_generator` usage for all random timesteps/noise.
  The base method explicitly requires avoiding global RNG state.
- Relevant lessons for future implementation:
  - `.agents/lessons/2026-05-07_dit-dtype-boundary-with-flash-attn.md`
  - `.agents/lessons/2026-05-07_silent-channel-major-packing-bugs.md`
  - `.agents/lessons/2026-05-07_conversion-cast-bf16-suffix-allowlist.md`
  These are mostly model-port lessons, but they are still relevant if TDM work
  introduces checkpoint conversion, LoRA export, or new latent/token handling.

## Suggested Next Steps

1. Confirm intended scope with maintainers: "TDM training method for Wan" versus
   "support official CogVideoX TDM LoRA" versus "documentation/plan only".
2. If native Wan TDM is desired, prototype `TDMMethod` next to `DMD2Method`.
   Keep it text-only first.
3. Translate the official demo's primitives into FastVideo abstractions:
   generated trajectory rollout, between-timestep noising, mixed-noise
   weighting, fake-score loss, and generator cooperative target.
4. Add a small fake-model/unit test for shape, timestep sampling, RNG
   determinism, and optimizer cadence without loading checkpoints.
5. Add an example config under
   `examples/train/configs/distribution_matching/wan/`.
6. Validate on Modal L40S through
   `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`.
   Do not run tests locally.

## Validation Status

- No code implementation was attempted in this pass.
- No local tests were run, consistent with repo instructions.
- This handoff is the only intended file change for the exploration branch.

## Add-Model Skill Phase 0 Gate

Generated: 2026-06-30 after explicit `/add-model` invocation.

Skill resources read:

- `.agents/skills/add-model/SKILL.md`
- `.agents/skills/add-model/shared/common_rules.md`
- `.agents/skills/add-model/contracts/prep_handoff.md`
- `.agents/skills/add-model/contracts/port_state.md`
- `.agents/skills/add-model/contracts/escape_hatch.md`
- `.agents/skills/add-model/contracts/component_context.md`
- `.agents/skills/add-model/contracts/parity_status.md`
- `.agents/skills/add-model/contracts/conversion_request.md`
- `.agents/skills/add-model/contracts/conversion_handoff.md`
- `.agents/skills/add-model/contracts/component_skill_handoff.md`
- `.agents/skills/add-model/contracts/pipeline_context.md`
- `.agents/skills/add-model/contracts/pipeline_handoff.md`
- `.agents/skills/add-model/contracts/final_handoff.md`
- `.agents/skills/add-model-01-prep/SKILL.md`

Live GitHub state was re-checked with `gh` as `macthecadillac`:

- `gh auth status --hostname github.com` reported active account
  `macthecadillac`.
- `gh issue view 775 --repo hao-ai-lab/FastVideo --comments ...` showed no new
  comments since the stale bot comment on 2026-05-31.
- `gh pr list --repo hao-ai-lab/FastVideo --state open --limit 100 ...` showed
  no active PR that directly targets TDM or issue 775.

Phase 0 result: **blocked before model/component work**.

Reasons:

1. This file is an exploration handoff, not an `add-model-01-prep` handoff and
   not an equivalent match for `contracts/prep_handoff.md`.
2. Required prep fields are missing or unresolved:
   `model_family`, `workload_types`, `official_ref_dir`,
   `official_ref_commit`, `hf_weights_path`, `hf_revision`,
   `local_weights_dir`, `source_layout`, `model_index_class`,
   `components_seen`, `needs_conversion`, `hf_token_env`,
   `dependency_changes`, `official_env_status`, `local_tests_readme`,
   `port_state_file`, `gitignore_entries_added`, `next_step`, and
   `escape_hatch`.
3. Required shared state files do not exist yet:
   `tests/local_tests/<model_family>/README.md` and
   `tests/local_tests/<model_family>/PORT_STATUS.md`.
4. Scope does not yet match the `add-model` skill's two supported shapes. The
   issue asks for TDM, which is a training/distillation method. It is not yet
   scoped as either a full model family/variant port or a first-class reusable
   component contribution.
5. The official TDM sources expose multiple possible directions: native Wan TDM
   training, official CogVideoX-2B LoRA support, SD3/SD3.5 image LoRA support,
   or planning/docs only. The first-PR variant and modality axes are therefore
   not locked.

Required next decision before any `/add-model` implementation phase:

- Either reroute this as a training-method feature outside `/add-model`, likely
  under `fastvideo/train/methods/distribution_matching/`, or provide the inputs
  needed to run `add-model-01-prep` for a concrete model/component scope.

If the next step is `add-model-01-prep`, the workflow requires these inputs:

1. Official reference repo or Diffusers pipeline URL.
2. HF repo id or local weights path, and whether it has a root
   `model_index.json`.
3. Target `model_family`.
4. Workload types.
5. Which token env var is exported: `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or
   `HF_API_KEY` (env var name only, no token value).
6. Approval to stage clone and weights under the FastVideo repo root.
7. Approval to install official reference dependencies into the current
   FastVideo environment for parity tests.

## TDM Integration Plan From Reference Repo

Generated: 2026-06-30 after user confirmed the reference implementation:
https://github.com/Luo-Yihong/TDM

Reference repo status:

- Repo was re-read through `gh` as `macthecadillac`.
- Current layout is still minimal: `README.md`, `requirements.txt`, `assets/`,
  and `train_tdm_demo.py`.
- The official training demo is PixArt/image-oriented diffusers+accelerate
  code, not a FastVideo-ready video trainer.
- The README shows pre-trained TDM LoRAs for SD3.5, SD3, Dreamshaper, and
  CogVideoX-2B. FastVideo does not currently expose a first-class CogVideoX
  pipeline in this branch, so importing the official CogVideoX LoRA is not the
  most direct native integration.

Recommended first PR scope:

- Implement **native TDM training for Wan T2V** in the modular training stack.
- Treat this as a training-method feature, not an `/add-model` model/component
  port.
- First target: Wan 2.1 T2V 1.3B, text-only/data-free, 4-step student, optional
  LoRA on the student.
- Defer TSAM/multi-NFE training until fixed-step TDM is validated.
- Defer official CogVideoX LoRA import unless a CogVideoX pipeline becomes a
  separate accepted scope.

Concrete code work:

1. Add `fastvideo/train/methods/distribution_matching/tdm.py`.
   - Either subclass `DMD2Method` for role validation and optimizer plumbing, or
     make a sibling `TDMMethod(TrainingMethod)` and copy only the shared
     optimizer/backward cadence.
   - Required roles should match DMD2: trainable `student`, frozen `teacher`,
     trainable `critic`/fake-score model.
2. Export `TDMMethod`.
   - Update `fastvideo/train/methods/distribution_matching/__init__.py`.
   - Update lazy exports in `fastvideo/train/methods/__init__.py` if desired for
     consistency.
3. Translate the official demo primitives into FastVideo model wrappers:
   - `generate_new(...)` -> student trajectory rollout over configured
     `tdm_denoising_steps`.
   - `Predictor.predict(...)` -> calls to `ModelBase.predict_noise` /
     `ModelBase.predict_x0`, preserving conditional/unconditional CFG behavior.
   - `Predictor.add_noise(samples, noise, t1, t2)` -> a scheduler helper for
     adding noise between two timesteps, not only from clean x0 to one timestep.
   - `Predictor.obtain_mixed_noise(...)` -> FastVideo scheduler-space mixed
     noise used for fake-score importance weighting.
4. Implement TDM critic/fake-score update:
   - Roll out the student trajectory with no grad.
   - Sample trajectory index `k`, target timestep `t_k`, midpoint/start
     timestep, and fake-score timestep using `TrainingMethod.cuda_generator`.
   - Build noised generated latents via the between-timestep noise helper.
   - Compute teacher prediction and critic prediction at the fake-score point.
   - Match official loss shape: fake-score x0/latent MSE weighted by SNR and
     importance ratio.
5. Implement TDM generator update:
   - Reuse the same trajectory/timestep sampling path.
   - Query frozen teacher conditional/unconditional outputs and critic output.
   - Construct the cooperative target equivalent to
     `model_latents + (teacher - fake) + cfg_delta`.
   - Support Huber option and normalization/weighting factor.
6. Add config knobs under `method`, not model config:
   - `tdm_denoising_steps` or reuse `dmd_denoising_steps`.
   - `total_train_timesteps`.
   - `generator_update_interval`.
   - `real_score_guidance_scale` / `cfg`.
   - `use_huber`, `huber_c`.
   - `noise_interval_mode`: `separate` versus `to_terminal`.
   - `use_randmid`.
   - fake-score optimizer knobs matching DMD2.
7. Add an example config:
   - `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`
   - Use `models.student.lora` for LoRA-first training.
   - Use existing Wan DMD validation pipeline with explicit 4-step sampling
     timesteps.
8. Update docs:
   - Add a TDM section to `docs/training/train_infra.md`.
   - Mention that the first integration is Wan-native TDM, not the official
     CogVideoX LoRA inference artifact.

Testing/validation plan:

- Add lightweight unit tests that do not load Wan weights:
  - config parsing/build target for `TDMMethod`;
  - timestep/trajectory sampling determinism through `cuda_generator`;
  - between-timestep noise helper shape and finite values using a fake scheduler;
  - optimizer cadence: critic every step, student according to
    `generator_update_interval`;
  - loss-map keys and backward routing with fake role models.
- Add/extend Modal validation:
  - dry-run config build on L40S;
  - short Wan TDM smoke on L40S using a tiny prompt set;
  - compare validation samples/loss curves against DMD2 baseline enough to catch
    obvious divergence.
- Do not claim quality parity from unit tests. A real Modal run is required for
  this method.

Main risks:

- The official script uses DDPM alpha/sigma helpers. Wan uses flow matching
  scheduler conventions, so the between-timestep noising and mixed-noise math
  must be rederived against `WanModel.noise_scheduler` instead of copied.
- The official video evidence is CogVideoX LoRA inference, not a full video
  training reference. Wan TDM is an adaptation of the algorithm, not a
  checkpoint-level port.
- TDM is more memory-expensive than DMD2 because each step may need student,
  teacher, and critic predictions on generated trajectory points. Start with
  LoRA and small validation cadence.

## Flux2 PR 1349 Pattern Applied To TDM

Generated: 2026-07-01 after inspecting merged PR #1349:
https://github.com/hao-ai-lab/FastVideo/pull/1349

What PR #1349 establishes as the local pattern:

- A model/workload integration is not just code. It includes a reviewer-facing
  local-test README, explicit reference assets, exact env vars, skip activation
  commands, a component/test matrix, and Modal evidence.
- Runtime code must be FastVideo-native. Third-party model classes belong in
  tests or passthrough loaders only when intentionally documented.
- Parity tests may skip in CI when weights/CUDA are absent, but the PR must
  record non-skip Modal evidence before claiming correctness.
- Shape-only tests are not enough. Flux2 replaced shape-only DiT checks with
  strict tensor comparisons and pipeline latent parity.
- The Flux2 PR decomposed coverage into:
  - typed-surface/preflight tests;
  - component parity;
  - pipeline smoke;
  - pipeline parity;
  - example execution;
  - CI quality placeholder/SSIM seeding follow-up.
- Its `tests/local_tests/flux2/README.md` is the best template for TDM's local
  evidence log.

Important mismatch:

- Flux2 is a model + pipeline port. TDM is a training/distillation method.
  Therefore, copy the validation discipline, not the pipeline/component phases.
  TDM should not add new inference registry/preset/pipeline code in the first
  PR unless the scope changes.

Detailed implementation plan:

### Phase A: Setup And State Tracking

1. Create a training-method local test area:
   - `tests/local_tests/tdm/README.md`
   - optionally `tests/local_tests/tdm/PORT_STATUS.md` if following the
     add-model state-file convention for resumability.
2. Use the Flux2 README layout:
   - model family: `tdm`
   - workload: training/distillation, first target `Wan2.1 T2V 1.3B`
   - official reference: `https://github.com/Luo-Yihong/TDM`
   - official reference file: `train_tdm_demo.py`
   - reference status: single-script PixArt/diffusers algorithm reference
   - source layout: `single_script_reference`
   - first FastVideo target: Wan-native modular training method
   - exact env vars for activated tests, e.g. `TDM_REF_DIR`,
     `TDM_WAN_MODEL_DIR`, and token env var name only if needed.
3. Clone the reference repo only for tests/review reproduction if needed.
   Production code must not import from it.
4. Add `.gitignore` entries only for staged reference/weight artifacts, not for
   source files.

### Phase B: Reference Decomposition

Translate the official demo into named algorithm units before writing FastVideo
code:

1. `generate_new(...)`:
   - Starts from pure noise.
   - Runs the student for `K` denoising steps.
   - Stores predicted clean latents and noised trajectory points.
2. `Predictor.predict(...)`:
   - Runs a score model at a noisy point.
   - Applies optional classifier-free guidance.
   - Converts predicted score/noise to predicted clean latent.
3. `Predictor.add_noise(samples, noise, t1, t2)`:
   - Adds additional noise from an intermediate timestep to a later/noisier
     timestep.
4. `Predictor.obtain_mixed_noise(...)`:
   - Computes the effective mixed noise for the importance ratio.
5. Fake-score update:
   - Samples a trajectory point.
   - Creates a noised generated latent.
   - Trains fake-score model to predict the generated clean latent.
   - Applies SNR clipping and importance weighting.
6. Generator update:
   - Queries teacher conditional/unconditional predictions and fake-score
     prediction.
   - Builds cooperative target:
     `student_x0 + (teacher_cond - fake) + (cfg - 1) * (teacher_cond - teacher_uncond)`.
   - Uses normalized MSE or Huber.

Record these six units in `tests/local_tests/tdm/README.md` so reviewers can
map code back to the reference script.

### Phase C: FastVideo Method Implementation

1. Add `fastvideo/train/methods/distribution_matching/tdm.py`.
2. Prefer `class TDMMethod(DMD2Method)` but override `single_train_step`.
   Reuse from `DMD2Method`:
   - role validation (`student`, `teacher`, `critic`);
   - CFG-uncond parsing;
   - student/critic optimizer setup;
   - optimizer/scheduler selection;
   - backward routing convention using `_fv_backward`.
3. Do not reuse DMD2's loss internals directly:
   - replace `_critic_flow_matching_loss` with `_tdm_fake_score_loss`;
   - replace `_dmd_loss` with `_tdm_generator_loss`;
   - replace `_student_rollout` with trajectory-producing rollout context.
4. Add helper dataclasses inside `tdm.py`:
   - `TDMTrajectory`: generated clean latents, noisy trajectory points,
     per-step timesteps/sigmas, selected index.
   - `TDMSampleContext`: selected noisy input, `t_g`, `t_mid`, sampled fake
     timestep, model prediction, random noise, mixed noise.
5. Add method config parsing with explicit errors:
   - `tdm_denoising_steps` (first PR fixed 4-step schedule);
   - `generator_update_interval`;
   - `real_score_guidance_scale` or `cfg`;
   - `fake_score_learning_rate`, `fake_score_betas`,
     `fake_score_lr_scheduler`;
   - `noise_interval_mode`: `separate` or `to_terminal`;
   - `use_randmid`;
   - `use_huber`, `huber_c`;
   - `snr_clip`, default matching the reference's `5`.
6. Export the method:
   - `fastvideo/train/methods/distribution_matching/__init__.py`
   - `fastvideo/train/methods/__init__.py`

### Phase D: Scheduler Translation

This is the highest-risk part.

Reference DDPM formula uses alpha/sigma. Wan `FlowMatchEulerDiscreteScheduler`
uses:

```text
x_t = (1 - sigma_t) * x0 + sigma_t * eps
```

Implement a flow-matching equivalent of reference
`Predictor.add_noise(samples, noise, t1, t2)`:

```text
x_t1 = (1 - s1) * x0 + s1 * eps_model
x_t2 = ((1 - s2) / (1 - s1)) * x_t1 + beta * eps_rand
beta = sqrt(s2^2 - (((1 - s2) / (1 - s1)) * s1)^2)
mixed_eps = ((((1 - s2) / (1 - s1)) * s1) * eps_model + beta * eps_rand) / s2
```

Guardrails:

- Only sample intervals where `s2 >= s1`; otherwise `beta` becomes invalid.
- Clamp tiny negative values before `sqrt` only when within numerical epsilon;
  treat larger negatives as a bug.
- Support `[B, T, C, H, W]` by flattening/unflattening the same way
  `ModelBase.predict_x0` does.
- Unit-test the formula with fake sigmas before any Wan run.

### Phase E: Training Config And Docs

1. Add example config:
   `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`.
2. First config should be LoRA-first:
   - `models.student.lora.enable: true`
   - explicit rank/alpha/target modules
   - teacher and critic initialized from the same Wan checkpoint; teacher frozen.
3. Use text-only/data-free prompt data if available. If existing text-only DMD
   data paths are used, document the exact expected preprocessing.
4. Add a section to `docs/training/train_infra.md` next to DMD2 and
   Self-Forcing:
   - what TDM does;
   - required roles;
   - key method knobs;
   - known limitation: first implementation is Wan-native, not official
     CogVideoX LoRA inference support.

### Phase F: Tests, Flux2-Style

Add tests before Modal validation:

1. `tests/local_tests/tdm/test_tdm_scheduler_math.py`
   - fake scheduler with known sigmas;
   - between-timestep noising produces finite tensors;
   - mixed noise reconstructs `x_t2`;
   - invalid interval raises.
2. `tests/local_tests/tdm/test_tdm_method_unit.py`
   - fake role models implementing `prepare_batch`, `add_noise`,
     `predict_noise`, `predict_x0`, and `backward`;
   - checks loss-map keys:
     `total_loss`, `generator_loss`, `fake_score_loss`;
   - checks critic backward every step and student backward only on
     `generator_update_interval`;
   - checks deterministic timestep sampling with `cuda_generator`.
3. `tests/local_tests/tdm/test_tdm_reference_math.py`
   - optional CPU-only parity against a minimal extracted version of the
     reference demo formulas using tiny tensors;
   - this is the closest analog to Flux2 component parity for a method.
4. `tests/local_tests/tdm/README.md`
   - copy Flux2's evidence style: commands, skip conditions, status table,
     Modal run IDs, and exact blockers.
5. Avoid package-level CI quality tests until there is a stable trained TDM
   artifact or a clear regression metric. A placeholder is acceptable only if it
   names the seeding/validation workflow and does not pretend to pass.

### Phase G: Modal Validation

Follow Flux2's evidence standard:

1. Modal preflight:
   - import `TDMMethod`;
   - build config from `tdm_t2v_lora.yaml`;
   - instantiate role models enough to catch target/config errors.
2. Modal unit/local tests:
   - run `tests/local_tests/tdm/` on L40S.
3. Short training smoke:
   - tiny prompt set;
   - low max steps, e.g. 2-10;
   - `output_type=latent` validation if using a validation callback;
   - verify no NaNs and expected loss keys log.
4. Training sanity:
   - record `loss_score` / fake-score loss and generator loss magnitudes;
   - compare to DMD2 only as a sanity baseline, not as parity.
5. Optional quality probe:
   - generate a few 4-step samples from the trained LoRA checkpoint;
   - record videos/latents as artifacts, but do not claim quality until reviewed.

### Phase H: Review Checklist Before PR

- No runtime imports from the TDM reference repo.
- No diffusers/transformers model-class imports in production training code.
- All randomness uses `TrainingMethod.cuda_generator`.
- Method knobs live under `method`, not model config.
- Flow-matching scheduler math has isolated tests.
- `tests/local_tests/tdm/README.md` has activated Modal evidence, not only skip
  commands.
- Handoff/README explicitly says this is a Wan adaptation of TDM, not an exact
  CogVideoX checkpoint port.
- Pre-commit is run on changed files before committing.

## Diffusion-To-Flow Bridge Proposal

Generated: 2026-07-01 in response to the explicit concern that the original
TDM paper/demo targets diffusion/CogVideoX-2B while FastVideo's first target is
Wan flow matching.

Position:

- Treat this as an algorithm port, not as a checkpoint/model port.
- Preserve TDM's role structure and training objective:
  - few-step generator/student trajectory;
  - fake-score/critic learning from generated samples;
  - teacher real-score guidance;
  - cooperative target
    `x0_student + (x0_teacher_cond - x0_fake) + cfg_delta`.
- Replace only the stochastic-process parameterization:
  - TDM reference uses diffusion alpha/sigma or alpha-bar formulas.
  - Wan uses flow matching with
    `x_sigma = (1 - sigma) * x0 + sigma * eps`.

FastVideo grounding:

- `FlowMatchEulerDiscreteScheduler.add_noise(...)` implements
  `x_sigma = sigma * noise + (1 - sigma) * sample`.
- Wan training batch construction uses the same formula for noisy model input.
- FastVideo's clean-latent conversion for flow models uses
  `pred_x0 = x_sigma - sigma * pred_noise`.
- Self-Forcing already demonstrates that an iterative Wan rollout can carry the
  current effective noise forward via:
  `eps = (x_sigma - (1 - sigma) * pred_x0) / sigma`, then
  `x_next = (1 - sigma_next) * pred_x0 + sigma_next * eps`.

Bridge design:

1. Use sigma as the single time coordinate.
   - Convert sampled timesteps to scheduler sigmas.
   - Avoid alpha-bar math in production TDM code.
   - Only keep reference alpha/sigma math in optional tiny-tensor tests.
2. Define the forward/noising family for Wan TDM as:

   ```text
   q_sigma(x | x0, eps) = (1 - sigma) * x0 + sigma * eps
   ```

3. Use Wan-native clean prediction everywhere:

   ```text
   x0_hat(model, x_sigma, sigma) = x_sigma - sigma * model_output
   ```

   where `model_output` is the Wan velocity/noise-style output already consumed
   by FastVideo's `pred_noise_to_pred_video`.

4. For ordinary noising from clean latents, call the model wrapper's existing
   `add_noise(clean_latents, noise, timestep)` so shapes and sequence-parallel
   flattening stay consistent with Wan.

5. For TDM's key intermediate-to-later noising step, do not call a DDPM
   scheduler. Derive the equivalent flow transition.

   Given an existing point:

   ```text
   x_s1 = (1 - s1) * x0 + s1 * eps_model
   ```

   choose a later/noisier point `s2 >= s1` and fresh independent noise
   `eps_rand`. Reuse as much of the existing noise component as possible:

   ```text
   a = (1 - s2) / (1 - s1)
   beta_sq = s2^2 - (a * s1)^2
   beta = sqrt(beta_sq)
   x_s2 = a * x_s1 + beta * eps_rand
   mixed_eps = (a * s1 * eps_model + beta * eps_rand) / s2
   ```

   This guarantees:

   ```text
   x_s2 = (1 - s2) * x0 + s2 * mixed_eps
   ```

   and gives the fake-score loss the effective mixed noise required for TDM's
   importance weighting.

6. Guard the transition aggressively:
   - sample only `s2 >= s1`;
   - error on meaningfully negative `beta_sq`;
   - clamp `beta_sq` only for tiny floating-point negatives;
   - handle `sigma == 0` terminal cases separately, never by divide-by-zero.

7. Keep the sampler direction clear:
   - inference denoising moves from high sigma to low sigma;
   - TDM's fake-score noising step moves from a generated intermediate point to
     an equal-or-higher sigma point for critic training.
   - Name helper args `sigma_from` and `sigma_to` instead of `t1`/`t2` to avoid
     DDPM timestep ambiguity.

8. Use the same denoising grid as Wan inference for the first PR.
   - Start with a fixed four-step schedule to match the reference's few-step
     setting.
   - Later allow explicit `method.tdm.sigmas` or `method.tdm.timesteps` once the
     baseline works.

Validation plan for the bridge:

1. CPU math tests:
   - construct tiny `x0`, `eps_model`, `eps_rand`, `s1`, `s2`;
   - build `x_s1`;
   - compute `x_s2` and `mixed_eps`;
   - assert reconstruction equality:
     `x_s2 == (1 - s2) * x0 + s2 * mixed_eps`.
2. Wan-shape tests:
   - repeat the same checks on `[B, T, C, H, W]`;
   - verify flatten/unflatten behavior matches Wan wrappers.
3. Direction tests:
   - `s2 < s1` raises;
   - `s2 == s1` reconstructs with `beta == 0` up to tolerance;
   - terminal `s1 == 0` and high-sigma cases are finite.
4. Method tests:
   - fake models return deterministic outputs;
   - TDM fake-score loss uses `mixed_eps`;
   - generator loss uses Wan-native `x0_hat` predictions for student, teacher,
     and critic.
5. Modal smoke:
   - first target is "stable and finite" training on Wan, not quality parity
     with CogVideoX;
   - record loss keys and no-NaN evidence before any quality claim.

Open research risk:

- TDM's paper-level derivation is tied to diffusion notation, but the core
  distribution-matching idea only needs a valid forward corruption family and a
  model-to-clean conversion. The Wan bridge is mathematically consistent for
  the linear flow noising family, but empirical behavior is not guaranteed until
  Modal training smoke and qualitative probes are run.

## Implementation Update: First Wan TDM Method Pass

Generated: 2026-07-01 after implementing the first code pass on branch
`issue/775-tdm` in worktree `/tmp/fastvideo-worktrees/issue-775-tdm`.

Scope implemented:

- Added `fastvideo/train/methods/distribution_matching/tdm.py`.
- Registered `TDMMethod` in:
  - `fastvideo/train/methods/distribution_matching/__init__.py`
  - `fastvideo/train/methods/__init__.py`
- Added Wan LoRA example config:
  - `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`
- Added local tests and reviewer README:
  - `tests/local_tests/tdm/README.md`
  - `tests/local_tests/tdm/__init__.py`
  - `tests/local_tests/tdm/test_tdm_scheduler_math.py`
  - `tests/local_tests/tdm/test_tdm_method_unit.py`
- Updated `docs/training/train_infra.md` with TDM roles, config keys, and the
  diffusion-to-flow caveat.

Implementation details:

- `TDMMethod` subclasses `DMD2Method` to reuse:
  - `student` / `teacher` / `critic` role validation;
  - fake-score optimizer and scheduler construction;
  - generator update interval;
  - CFG-unconditional parsing;
  - custom student/critic backward routing.
- TDM currently requires `method.rollout_mode: simulate`.
- First target config uses LoRA for both student and critic, with frozen
  teacher.
- No production runtime imports from `Luo-Yihong/TDM`.

Flow bridge code added:

- `flow_effective_noise(...)`
  recovers effective Wan flow noise from
  `x_sigma = (1 - sigma) * x0 + sigma * eps`.
- `flow_snr(...)`
  uses the flow analogue `((1 - sigma) / sigma)^2`.
- `flow_transition_to_noisier_sigma(...)`
  implements the Wan equivalent of TDM's between-timestep noising:

  ```text
  a = (1 - sigma_to) / (1 - sigma_from)
  beta_sq = sigma_to^2 - (a * sigma_from)^2
  x_to = a * x_from + sqrt(beta_sq) * eps_rand
  mixed_eps = (a * sigma_from * eps_from + sqrt(beta_sq) * eps_rand) / sigma_to
  ```

- The transition rejects `sigma_to < sigma_from`, `sigma_from == 1`, and
  `sigma_to == 0`; it only clamps tiny negative `beta_sq` values caused by
  numerical noise.

TDM trajectory behavior:

- `_student_trajectory(...)` starts from pure noise and runs the student over
  `tdm_denoising_steps`.
- `student_sample_type: sde` re-noises each predicted `x0` with fresh noise at
  the next lower sigma.
- `student_sample_type: ode` carries effective flow noise to the next lower
  sigma, matching the pattern already used in Self-Forcing.
- `enable_gradient_in_rollout: false` defaults to DMD2-like memory behavior:
  only the final student prediction carries generator gradients.

TDM fake-score loss:

- Builds a no-grad student trajectory.
- Samples a generated trajectory point whose sigma is not the max sigma.
- Moves it to a strictly larger sigma when possible.
- Trains the critic through `critic.predict_x0(...)` to recover the generated
  clean latent.
- Applies flow-SNR clipping and Gaussian mixed-noise importance weighting:
  `exp(0.5 * (proposal_noise^2 - mixed_noise^2))`, clipped by
  `importance_weight_clip`.

TDM generator loss:

- Uses final generated clean latent from the student trajectory.
- Noises it at a sampled Wan training timestep.
- Queries:
  - critic fake `x0`;
  - teacher conditional `x0`;
  - teacher unconditional `x0`.
- Builds CFG teacher target as:
  `teacher_uncond + scale * (teacher_cond - teacher_uncond)`.
- Pushes the student toward:
  `student_x0 + (teacher_cfg_x0 - fake_x0)`, optionally normalized by
  `mean(abs(student_x0 - teacher_cfg_x0))`.

Local checks run:

```text
python -m py_compile fastvideo/train/methods/distribution_matching/tdm.py \
  tests/local_tests/tdm/test_tdm_scheduler_math.py \
  tests/local_tests/tdm/test_tdm_method_unit.py
```

Result: passed.

```text
uvx mypy --python-version 3.10 --follow-imports skip \
  --disable-error-code union-attr --disable-error-code override \
  --explicit-package-bases \
  fastvideo/train/methods/__init__.py \
  fastvideo/train/methods/distribution_matching/__init__.py \
  fastvideo/train/methods/distribution_matching/tdm.py
```

Result: passed.

Pre-commit status:

```text
uvx pre-commit run --files <changed-files>
```

Result:

- `yapf`: passed
- `ruff`: passed
- `codespell`: passed
- `PyMarkdown`: passed
- `actionlint`: skipped, no files
- `check-filenames`: passed
- `mypy`: failed before type-checking with
  `issue-775-tdm is not a valid Python package name`

The mypy failure is caused by the hyphenated worktree basename plus the repo's
tracked root `__init__.py`. Direct mypy with the same hook args and
`--explicit-package-bases` passes, as shown above.

Local tests intentionally not run:

- `pytest tests/local_tests/tdm/ -v -s`

Reason: repo AGENTS.md says not to run tests locally; validation should happen
on Modal.

Modal validation run:

```text
python -m modal run fastvideo/tests/modal/launch_l40s_job.py \
  --gpu-type L40S \
  --num-gpus 1 \
  --install-extra test \
  --git-commit 966682510cbebb87ac0dd29e0b3cbd26716bb091 \
  --command "pytest tests/local_tests/tdm/ -v -s"
```

Result:

```text
Modal app: ap-scjQBtLeENXy1GXPNSjtoP
tests/local_tests/tdm/test_tdm_method_unit.py::test_tdm_single_train_step_reports_losses_and_routes_backward PASSED
tests/local_tests/tdm/test_tdm_method_unit.py::test_tdm_respects_generator_update_interval PASSED
tests/local_tests/tdm/test_tdm_scheduler_math.py::test_flow_transition_reconstructs_noisier_point_for_video_latents PASSED
tests/local_tests/tdm/test_tdm_scheduler_math.py::test_flow_transition_raises_for_lower_sigma_target PASSED
tests/local_tests/tdm/test_tdm_scheduler_math.py::test_flow_effective_noise_and_snr_match_wan_parameterization PASSED
5 passed, 14 warnings in 16.94s
```

Known remaining work:

- Run a very short Wan TDM training smoke with
  `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`.
- Inspect loss keys and confirm no NaNs.
- Decide whether `enable_gradient_in_rollout` should remain default `false` or
  switch to `true` after memory and behavior are observed.
- Compare `student_sample_type: sde` versus `ode`; first config uses `sde`.
