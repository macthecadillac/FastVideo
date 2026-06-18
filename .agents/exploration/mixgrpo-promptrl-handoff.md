# Exploration Log: MixGRPO and PromptRL Integration

## Status: branch checkpoint

## Context

The current task is to integrate GenRL-style MixGRPO and PromptRL into the local
FastVideo modular training stack. The implementation should use FastVideo PR
#1450 as the foundation, with work on branch `feat-mixgrpo-promptrl`.

User constraints:

- All validation runs must use Modal via
  `fastvideo/tests/modal/launch_l40s_job.py`.
- Do not run heavyweight training/tests locally.
- Make commits when a coherent slice is validated.
- Keep handoff notes updated so interrupted work can resume quickly.

## Progress

- [x] Created branch `feat-mixgrpo-promptrl`.
- [x] Confirmed local `main` already contains PR #1450.
- [x] Read RL-related agent guidance:
  - `.agents/skills/add-rl-method/SKILL.md`
  - `.agents/skills/rlhf-training-abstractions/SKILL.md`
- [x] Started first infrastructure slice for MixGRPO:
  - extended `SamplingConfig` with `mixed_ode_sde`,
    `sde_window_start`, `sde_window_size`, and `sde_noise_scale`;
  - added `sde_step_mask`;
  - changed `DiffusionSampler` to use one trajectory path that performs ODE
    scheduler steps outside the configured SDE window and stochastic reflow
    inside it;
  - added shared GRPO advantage and clipped policy-loss helpers in
    `fastvideo/train/methods/rl/common/advantages.py`;
  - exported the new helpers from `fastvideo/train/methods/rl/common/__init__.py`;
  - added focused component tests in
    `tests/local_tests/test_train_rl_sampling.py` and
    `tests/local_tests/test_train_rl_advantages.py`.
- [x] Run the focused tests on Modal L40S using the local patch launcher.
- [x] Commit the first validated infrastructure slice.
- [x] Wire MixGRPO method behavior on top of the shared helpers.
- [x] Add the first PromptRL foundation slice for static prompt refinement.
- [x] Validate the current PromptRL model-policy boundary slice.
- [x] Commit the current PromptRL model-policy boundary slice.
- [x] Validate the concrete causal-LM prompt-refiner role slice.
- [x] Commit the concrete causal-LM prompt-refiner role slice.

## Current Working Tree Notes

Expected changed files since base `633d3935`:

- `examples/train/configs/rl/wan/mixgrpo_pick_clip.yaml`
- `fastvideo/tests/modal/launch_l40s_job.py`
- `fastvideo/train/methods/rl/__init__.py`
- `fastvideo/train/methods/rl/common/sampling.py`
- `fastvideo/train/methods/rl/common/advantages.py`
- `fastvideo/train/methods/rl/common/__init__.py`
- `fastvideo/train/methods/rl/common/prompt_refinement.py`
- `fastvideo/train/methods/rl/mix_grpo.py`
- `fastvideo/train/models/prompt_refiner.py`
- `tests/local_tests/test_train_rl_sampling.py`
- `tests/local_tests/test_train_rl_advantages.py`
- `tests/local_tests/test_mix_grpo_method_utils.py`
- `tests/local_tests/test_prompt_refinement.py`
- `tests/local_tests/test_prompt_refiner_model.py`
- `.agents/exploration/mixgrpo-promptrl-handoff.md`

Known unrelated file:

- `docker/Containerfile` is untracked and pre-existing. Do not touch it unless
  explicitly asked.

## Current Resume Point

Current branch head:

- `4f4c28be [feat]: add causal LM PromptRL refiner`

Prior commits on this branch:

- `7b0c8694 [feat]: add PromptRL prompt policy hooks`
- `49a7336e [feat]: add PromptRL prompt refinement foundation`
- `056cbeba [feat]: add MixGRPO training method`
- `e5b37ae8 [feat]: add MixGRPO RL helpers`

Current PromptRL policy hook slice:

- Extends `PromptRefinementConfig` with `mode: model`.
- Adds `PromptRefinementResult.policy_mask`, prompt-refiner log-prob fields,
  and per-sample metadata.
- Allows `refine_prompt_batch(..., prompt_refiner=...)` to call a
  duck-typed `prompt_refiner.refine_prompts(...)`, while preserving
  `num_original_prompts`.
- Lets `MixGRPOMethod` optionally use `models.prompt_refiner` and
  `models.old_prompt_refiner`.
- Adds optional prompt-refiner optimizer/scheduler wiring and a clipped
  GRPO-style prompt-policy loss using
  `prompt_refiner.compute_log_probs(...)`.
- Adds focused tests for model-mode prompt refinement and prompt-refiner
  optimizer stepping.
- Modal focused tests passed for this slice:
  `40 passed, 14 warnings in 18.37s`.
- Modal pre-commit passed for this slice:
  yapf, ruff, codespell, mypy, filename-spaces, and suggestion hooks all passed.
- Committed as:
  `7b0c8694 [feat]: add PromptRL prompt policy hooks`.

Causal-LM prompt-refiner slice:

- Adds `fastvideo.train.models.prompt_refiner.CausalLMPromptRefiner`, a
  `ModelBase` role wrapper around an HF causal LM.
- The wrapper implements `refine_prompts(...)` for rollout and
  `compute_log_probs(...)` for PromptRL policy optimization.
- The log-prob path masks only generated/refined response tokens, not the
  instruction/original-prompt prefix.
- Adds `tests/local_tests/test_prompt_refiner_model.py` with a fake tokenizer
  and fake LM, avoiding HF downloads.
- Adds commented PromptRL role guidance to
  `examples/train/configs/rl/wan/mixgrpo_pick_clip.yaml`.
- New files were temporarily marked with `git add -N` so Modal patch export
  included them before commit.
- First Modal run for this slice failed due a test-fixture bug:
  `_CharTokenizer` emitted token id 9 while `_UniformLM(vocab_size=8)` was too
  small. Fixed by using vocab size 16 in that test.
- Modal focused tests passed after the fixture fix:
  `43 passed, 14 warnings in 18.04s`.
- Modal pre-commit passed:
  yapf, ruff, codespell, mypy, filename-spaces, and suggestion hooks all passed.
- Committed as:
  `4f4c28be [feat]: add causal LM PromptRL refiner`.

Last focused Modal test command:

```bash
/home/toolbox/venv/bin/python -m modal run fastvideo/tests/modal/launch_l40s_job.py \
  --command "pytest tests/local_tests/test_train_rl_sampling.py tests/local_tests/test_train_rl_advantages.py tests/local_tests/test_mix_grpo_method_utils.py tests/local_tests/test_prompt_refinement.py tests/local_tests/test_prompt_refiner_model.py -q" \
  --gpu-type L40S \
  --num-gpus 1 \
  --git-commit 633d39356804e63478d242611e992dc8e1af3caa \
  --apply-local-patch \
  --patch-base 633d39356804e63478d242611e992dc8e1af3caa \
  --patch-paths "examples/train/configs/rl/wan/mixgrpo_pick_clip.yaml,fastvideo/tests/modal/launch_l40s_job.py,fastvideo/train/methods/rl/__init__.py,fastvideo/train/methods/rl/common/__init__.py,fastvideo/train/methods/rl/common/advantages.py,fastvideo/train/methods/rl/common/prompt_refinement.py,fastvideo/train/methods/rl/common/sampling.py,fastvideo/train/methods/rl/mix_grpo.py,fastvideo/train/models/prompt_refiner.py,tests/local_tests/test_mix_grpo_method_utils.py,tests/local_tests/test_prompt_refinement.py,tests/local_tests/test_prompt_refiner_model.py,tests/local_tests/test_train_rl_advantages.py,tests/local_tests/test_train_rl_sampling.py"
```

Last remote pre-commit command:

```bash
/home/toolbox/venv/bin/python -m modal run fastvideo/tests/modal/launch_l40s_job.py \
  --command "pre-commit run --files examples/train/configs/rl/wan/mixgrpo_pick_clip.yaml fastvideo/tests/modal/launch_l40s_job.py fastvideo/train/methods/rl/__init__.py fastvideo/train/methods/rl/common/__init__.py fastvideo/train/methods/rl/common/advantages.py fastvideo/train/methods/rl/common/prompt_refinement.py fastvideo/train/methods/rl/common/sampling.py fastvideo/train/methods/rl/mix_grpo.py fastvideo/train/models/prompt_refiner.py tests/local_tests/test_mix_grpo_method_utils.py tests/local_tests/test_prompt_refinement.py tests/local_tests/test_prompt_refiner_model.py tests/local_tests/test_train_rl_advantages.py tests/local_tests/test_train_rl_sampling.py" \
  --gpu-type L40S \
  --num-gpus 1 \
  --git-commit 633d39356804e63478d242611e992dc8e1af3caa \
  --apply-local-patch \
  --patch-base 633d39356804e63478d242611e992dc8e1af3caa \
  --patch-paths "examples/train/configs/rl/wan/mixgrpo_pick_clip.yaml,fastvideo/tests/modal/launch_l40s_job.py,fastvideo/train/methods/rl/__init__.py,fastvideo/train/methods/rl/common/__init__.py,fastvideo/train/methods/rl/common/advantages.py,fastvideo/train/methods/rl/common/prompt_refinement.py,fastvideo/train/methods/rl/common/sampling.py,fastvideo/train/methods/rl/mix_grpo.py,fastvideo/train/models/prompt_refiner.py,tests/local_tests/test_mix_grpo_method_utils.py,tests/local_tests/test_prompt_refinement.py,tests/local_tests/test_prompt_refiner_model.py,tests/local_tests/test_train_rl_advantages.py,tests/local_tests/test_train_rl_sampling.py"
```

If network/auth sandboxing blocks a Modal run, retry the same command with
approval escalation. The user explicitly approved sending local patches and
remote Modal actions.

Current Modal client status:

- `modal run ...` failed locally because no `modal` executable is on `PATH`.
- `python -m modal ...` is the correct launcher form from the script header, but
  the active Python environment initially reported `No module named modal`.
- Installed the Modal Python client into `/home/toolbox/venv` with
  `uv pip install modal`.
- Attempted to submit the focused test command through
  `fastvideo/tests/modal/launch_l40s_job.py` with `--apply-local-patch`.
- The command was blocked by the approval reviewer because `--apply-local-patch`
  sends local workspace diffs to Modal, an external service.
- User explicitly approved sending local patches to the remote Modal machine and
  pre-approved actions needed on the remote machine.
- First retry with `--apply-local-patch` started the Modal app, but failed before
  remote execution because `--patch-paths` was passed as a space-separated list.
  The launcher expects comma-separated paths.
- Retry with comma-separated paths reached remote pytest:
  `3 failed, 23 passed`.
- Failures were in focused tests rather than remote setup:
  - mixed sampler test inspected `model.noise_scheduler.step_calls`, but
    `DiffusionSampler` deep-copies the scheduler for `model_default`;
  - weighted reward-first assertion tolerance was too tight for fp32 values with
    `epsilon=1e-6`;
  - weighted-advantages expected vector was wrong for the first prompt group.
- Fixed those tests locally:
  - mixed sampler now asserts fake-model `predict_noise`/`predict_x0` call
    counts;
  - weighted reward-first tolerance is `3e-6`;
  - weight-advantages expected vector is `[-0.5, 0.5, -0.5, 0.5]`.
- Second Modal run reached `25 passed, 1 failed`. The remaining failure was the
  weighted reward-first tolerance still at `2e-6` in the untracked test file.
- Fixed that tolerance to `3e-6`.
- Third Modal run passed:
  `26 passed, 14 warnings in 13.88s`.
- Remote pre-commit through `fastvideo/tests/modal/launch_l40s_job.py` passed:
  yapf, ruff, codespell, mypy, filename-spaces, and suggestion hooks all passed.
- Committed the first MixGRPO infrastructure slice:
  `e5b37ae8 [feat]: add MixGRPO RL helpers`.
- Next step is to start the next implementation slice: wire a MixGRPO method on
  top of the shared helpers, with PromptRL still reserved for the later stage.
- Started next slice design:
  - keep DiffusionNFT behavior unchanged;
  - add a sampler trace/log-prob path for stochastic SDE-window transitions;
  - add `MixGRPOMethod` as a DiffusionNFT-based subclass that overrides
    sampling, grouped multi-reward advantages, and PPO/GRPO clipped policy loss;
  - add a Wan MixGRPO example config and fake-model tests.
- Added `--patch-base` to `fastvideo/tests/modal/launch_l40s_job.py` because
  the first local commit is not pushed to GitHub. Modal can now checkout
  `633d3935` and receive the full local branch delta via
  `git diff --binary <patch_base>`.
- Second-slice Modal tests passed through `launch_l40s_job.py` using
  `--git-commit 633d3935 --patch-base 633d3935`:
  `32 passed, 14 warnings in 25.33s`.
- First remote pre-commit attempt had only yapf modifications; ruff, codespell,
  mypy, filename-spaces, and suggestion hooks passed.
- Retrieved formatter-only diffs through Modal and applied them locally:
  one line wrap in `common/sampling.py`, one line wrap in `mix_grpo.py`.
- Full remote pre-commit passed after formatting:
  yapf, ruff, codespell, mypy, filename-spaces, and suggestion hooks all passed.
- Committed the MixGRPO method wiring slice:
  `056cbeba [feat]: add MixGRPO training method`.
- Next stage is PromptRL. Start by adding prompt-selection/reweighting
  infrastructure that can run before sampling without disturbing the MixGRPO
  policy objective.
- PromptRL scope for first slice:
  - add reusable prompt refinement helpers under `rl/common`;
  - support disabled/no-op, dataset-column rewrites, and deterministic template
    rewrites;
  - preserve a configurable number of original prompts per local batch for the
    partial-refinement pattern described by PromptRL/UniRL;
  - integrate the helper into MixGRPO sampling only, leaving trainable LM
    optimization as a later slice.
- PromptRL foundation tests passed through Modal:
  `37 passed, 14 warnings in 19.57s`.
- PromptRL foundation remote pre-commit passed:
  yapf, ruff, codespell, mypy, filename-spaces, and suggestion hooks all passed.
- Committed the PromptRL foundation slice:
  `49a7336e [feat]: add PromptRL prompt refinement foundation`.
- Started the next PromptRL slice:
  - added a PromptRL model-refiner boundary with partial-refinement support;
  - added optional MixGRPO prompt-refiner optimization hooks;
  - Modal pytest and pre-commit validation passed;
  - committed as `7b0c8694`.

## Findings

- PR #1450 already provides the right training-method boundary for this work:
  method-managed RL outer loop, shared RL sampler, reward framework, and
  fake-model local component tests.
- GenRL-style MixGRPO needs two reusable pieces before a full method:
  prompt-grouped multi-reward advantage normalization and a mixed ODE/SDE
  sampling window.
- `mixed_ode_sde` should not just alias full SDE reflow. The sampler now honors
  the configured SDE window and uses ODE scheduler steps elsewhere.

## Mistakes / Dead Ends

- A weighted-reward-first test initially expected `[0, 0, -1, 1]`, but the
  weighted rewards normalize to `[-1, 1, -1, 1]` by prompt group. The test was
  corrected before validation.
- The first `mixed_ode_sde` config change accepted the new trajectory before the
  sampler honored it. The sampler path was updated before running Modal.

## Proposed Standardization

If this integration path works, promote the recurring pattern into a short SOP:

- add shared RL math/helper primitives first;
- validate with fake-model Modal tests;
- commit the helper slice;
- wire the full RL method;
- then add higher-level PromptRL behavior.
