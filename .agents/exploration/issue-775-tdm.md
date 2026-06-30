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
