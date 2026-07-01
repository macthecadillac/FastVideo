# Issue 775 TDM Handoff

Compacted: 2026-07-01

This file intentionally replaces the earlier long chronological log. Older
per-run details are preserved in branch commits and Modal artifact paths; this
handoff keeps only state needed to continue work.

## Worktree

- Repo: `github.com/hao-ai-lab/FastVideo`
- Branch: `issue/775-tdm`
- Worktree: `/tmp/fastvideo-worktrees/issue-775-tdm`
- Modal launcher worktree: `/tmp/fastvideo-worktrees/interleavethinker-modal`
  on branch `interleavethinker`
- Handoff: `.agents/exploration/issue-775-tdm.md`
- Current latest commit before compaction: `008883d9`

## GitHub State

Last checked: 2026-07-01 with `gh` authenticated as `macthecadillac`.

- Issue: https://github.com/hao-ai-lab/FastVideo/issues/775
- Title: `[Feature] TDM`
- State: open
- Assignee: `macthecadillac`
- Labels: `good first issue`, `contribution-needed`, `stale`, `keep-open`
- Comments: only the 2026-02-04 maintainer interest comment and the
  2026-05-31 stale-bot comment.
- Open PR list: no active PR directly targeting TDM or issue 775.

## Scope Decision

- Treat this as a **Wan-native TDM training-method integration**, not an
  `/add-model` model/component port.
- First target: Wan 2.1 T2V 1.3B, text-only/data-free, LoRA-first.
- Production code must not import from `Luo-Yihong/TDM`.
- Keep official CogVideoX LoRA import and TSAM/multi-NFE support out of the
  first scope.

## Reference Summary

Reference sources read:

- Paper: `Learning Few-Step Diffusion Models by Trajectory Distribution
  Matching`, arXiv:2503.06674
- Project: https://tdm-t2x.github.io/
- Repo: https://github.com/Luo-Yihong/TDM
- Script: `train_tdm_demo.py`

Important reference behavior:

- Roles are trainable generator/student, frozen teacher, trainable fake-score
  model.
- Generated trajectory is built under `torch.no_grad()`.
- Generator update samples from that no-grad trajectory and recomputes one
  student prediction for the generator loss.
- No reference equivalent exists for a user-facing full-rollout gradient
  option.
- The reference is diffusion/DDPM-style; FastVideo Wan uses flow matching.

## Implementation State

Key files:

- `fastvideo/train/methods/distribution_matching/tdm.py`
- `fastvideo/train/methods/distribution_matching/__init__.py`
- `fastvideo/train/methods/__init__.py`
- `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`
- `tests/local_tests/tdm/`
- `docs/training/train_infra.md`

Implemented behavior:

- `TDMMethod` subclasses `DMD2Method` to reuse role validation, optimizers,
  fake-score optimizer plumbing, CFG parsing, and backward routing.
- Required roles: `student`, `teacher`, `critic`.
- Required rollout mode: `simulate`.
- Flow bridge uses Wan noising:

  ```text
  x_sigma = (1 - sigma) * x0 + sigma * eps
  ```

- Between-sigma transition helper:

  ```text
  a = (1 - sigma_to) / (1 - sigma_from)
  beta_sq = sigma_to^2 - (a * sigma_from)^2
  x_to = a * x_from + sqrt(beta_sq) * eps_rand
  mixed_eps = (a * sigma_from * eps_from + sqrt(beta_sq) * eps_rand) / sigma_to
  ```

- `noise_interval_mode: separate` samples strictly larger non-terminal target
  sigmas only.
- `noise_interval_mode: to_terminal` remains explicit terminal behavior.
- `student_sample_type` supports `sde` and `ode`; first config stays `sde`.
- TDM no longer exposes `enable_gradient_in_rollout`; reference-style
  final-prediction-only gradients are unconditional.
- JSONL tracker support exists for persisted scalar diagnostics.

## Key Commits

- `0ebbcbf5` `[feat]: add JSONL training tracker`
- `ca654254` `[fix]: avoid terminal sigma in TDM separate noising`
- `1afc867c` `[fix]: remove TDM rollout gradient option`
- `008883d9` `[misc]: record TDM ode rollout diagnostic`

## Validation Summary

Local machine:

- Do not run pytest locally. Repo guidance says validation should run on Modal.
- Local syntax/type-oriented checks used so far:
  - `python -m py_compile ...`
  - `git diff --check`
  - `uvx pre-commit run --files ...`
  - direct `uvx mypy ... --explicit-package-bases ...`
- Pre-commit's mypy hook fails before type-checking in this worktree because
  `issue-775-tdm` is not a valid Python package name. Direct mypy passes.

Modal smoke/unit evidence:

- App `ap-QFB1GdYJkjQDHriHhY9Oor`
- Commit `1afc867c`
- Command: `pytest tests/local_tests/tdm/ -v -s`
- Result: `10 passed, 14 warnings`

20-step SDE diagnostic:

- App `ap-C7T7LCY4j0iGdCUigW85XX`
- Output root: `/root/data/tdm_diag_fake_score_fix_20_ca654254`
- Metrics: `/tmp/tdm_diag_fake_score_fix_20_ca654254_metrics.jsonl`
- Result: completed on 4x L40S with finite losses.
- No terminal fake-score targets, zero SNR weights, zero final weights, zero
  fake-score losses, or nonfinite scalar metrics.
- `update_student`: `[5, 10, 15, 20]`

20-step ODE diagnostic:

- App `ap-x6X9P0DJPpdV5LB5LIygWM`
- Output root: `/root/data/tdm_diag_ode_20_b55aeb32`
- Metrics: `/tmp/tdm_diag_ode_20_b55aeb32_metrics.jsonl`
- Result: completed on 4x L40S with finite losses.
- No terminal fake-score targets, zero SNR weights, zero final weights, zero
  fake-score losses, or nonfinite scalar metrics.
- ODE mean step time was lower than SDE in the 20-step diagnostic, but this is
  not quality evidence.

Current config decision:

- Keep `student_sample_type: sde` as first default because it more closely
  matches the upstream reference, which re-noises each predicted clean latent
  with model-predicted noise at the next timestep.
- Keep ODE as a viable alternative pending longer/qualitative evidence.

## Modal Data And Weights

Wan weights in Modal volume:

```text
/root/data/.cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
```

Small staged preprocessed T2V dataset:

```text
/root/data/.cache/datasets--wlsaidhi--crush-smol_processed_t2v/snapshots/67dd07f2163ad2b3397f8b3d8125b67ca452dc85/combined_parquet_dataset
```

Dataset summary from earlier Modal runs:

- 4 parquet shards
- 32 rows
- 8 rows per SP group in 4-GPU runs

## Current Plan

Goal: move beyond smoke testing and run a bounded quality/convergence pilot.

Run a 200-step Modal job with the current default `student_sample_type: sde`.

Why 200 steps:

- 20-step runs proved wiring and short-run stability.
- 200 steps is long enough to observe loss behavior and validation artifacts
  without jumping straight to an expensive full training run.
- 500 steps can be considered after the 200-step pilot looks healthy.

Pilot settings:

- GPU: 4x L40S
- Method: Wan TDM default config, `student_sample_type: sde`
- Max steps: `200`
- JSONL tracker: enabled
- Validation: every `100` steps
- DCP training-state checkpoints: every `100` steps
- Output root:
  `/root/data/tdm_pilot_sde_200_<shortsha>`
- Use `--commit-volume` so metrics, validation videos, and checkpoints persist.

Planned command shape after committing this compact handoff:

```text
python -m modal run fastvideo/tests/modal/launch_l40s_job.py \
  --gpu-type L40S \
  --num-gpus 4 \
  --install-extra test \
  --commit-volume \
  --git-commit <current-issue-branch-sha> \
  --command "torchrun --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml \
    --training.data.data_path /root/data/.cache/datasets--wlsaidhi--crush-smol_processed_t2v/snapshots/67dd07f2163ad2b3397f8b3d8125b67ca452dc85/combined_parquet_dataset \
    --training.loop.max_train_steps 200 \
    --training.checkpoint.training_state_checkpointing_steps 100 \
    --training.checkpoint.checkpoints_total_limit 3 \
    --training.checkpoint.output_dir /root/data/tdm_pilot_sde_200_<shortsha> \
    --training.tracker.trackers [jsonl] \
    --training.tracker.run_name tdm_pilot_sde_200 \
    --callbacks.validation.every_steps 100 \
    --callbacks.validation.output_dir /root/data/tdm_pilot_sde_200_<shortsha>/validation"
```

Pilot success checks:

- Job completes without OOM, worker restart, or distributed failure.
- Metrics JSONL exists and has 200 loss rows.
- `total_loss`, `fake_score_loss`, and `generator_loss` are finite.
- `fake_score_loss` does not collapse to all zeros.
- `update_student` follows `generator_update_interval=5`.
- Terminal-sigma diagnostics remain clean:
  - no `sigma_to_is_terminal`
  - no zero `snr_weight`
  - no zero `weight_mean`
- Validation MP4s exist at expected steps and decode successfully.
- Video frame count and resolution match the Wan validation config.
- Pixel statistics are not constant/all-black/all-white/corrupt.
- DCP checkpoint directories exist for expected saved steps.

What this still will not prove:

- TDM convergence.
- Flow-matching bridge equivalence to the paper in output quality.
- Whether the distilled model is good.
- Whether 4-step Wan TDM beats or matches DMD or Self-Forcing baselines.

## Current Remaining Work

1. Execute the 200-step SDE pilot above and record results here.
2. If healthy, compare validation artifacts visually and decide whether a
   500-step run or baseline DMD/Self-Forcing comparison is warranted.
3. Before any PR, re-check issue/PR state, rerun Modal tests, and prepare a
   draft PR only if explicitly requested.
