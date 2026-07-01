# Issue 775 TDM Handoff

Compacted: 2026-07-01
Last updated: 2026-07-01 after the 200-step SDE pilot

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
- Latest pushed issue-branch commit before the 200-step pilot:
  `4e7d9e76` `[misc]: compact TDM handoff and pilot plan`

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
- `4e7d9e76` `[misc]: compact TDM handoff and pilot plan`

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

200-step SDE pilot:

- App: `ap-jDoLULWN2Y6VACZJCHlCQr`
- Commit: `4e7d9e760df04c9e569a47215e74e84bbdf6a7dd`
- Output root: `/root/data/tdm_pilot_sde_200_4e7d9e76`
- Command summary: 4x L40S, Wan TDM default config, `student_sample_type: sde`,
  `max_train_steps=200`, JSONL tracker enabled, validation every 100 steps,
  DCP training-state checkpoints every 100 steps, `--commit-volume`.
- Result: completed successfully. Training progress reached `200/200` in
  about `1:17:55`; Modal volume commit completed.
- Non-fatal warnings observed:
  - `checkpoint-200` was saved at the step-200 checkpoint boundary and then
    saved again by final training teardown, producing PyTorch DCP overwrite
    warnings.
  - NCCL warned that `destroy_process_group()` was not called before process
    exit. The job still exited with code 0.
- Metrics file:
  `/tmp/tdm_pilot_sde_200_4e7d9e76_metrics.jsonl` downloaded from
  `/root/data/tdm_pilot_sde_200_4e7d9e76/tracker/metrics.jsonl`.
- Metrics summary:
  - JSONL rows: `640`
  - Unique steps: `1..200`
  - Loss rows: `200`
  - Critic grad rows: `200`
  - Student grad rows: `40`
  - `update_student` steps exactly matched `[5, 10, ..., 200]`
  - Nonfinite scalar metrics: `0`
  - `sigma_to_is_terminal` steps: `0`
  - zero `snr_weight` steps: `0`
  - zero `weight_mean` steps: `0`
  - zero `fake_score_loss` steps: `0`
  - `total_loss`: min `0.0004079751`, max `0.7137516737`,
    mean `0.0966718541`, first `0.0016110574`, last `0.6687002182`
  - `fake_score_loss`: min `0.0004079751`, max `0.0017997533`,
    mean `0.0010606790`, first `0.0016110574`, last `0.0012961911`
  - `generator_loss`: min `0.0`, max `0.7124048471`,
    mean `0.0956111750`, first `0.0`, last `0.6674040556`
  - `step_time_sec`: min `18.0589167890`, max `42.2056847560`,
    mean `23.0179435935`
- Tracker artifacts exist:
  - `tracker/config.json`
  - `tracker/metrics.jsonl`
  - `tracker/artifacts.jsonl`
  - `tracker/files`
- Checkpoints exist:
  - `checkpoint-100/{metadata.json,rng_state_rank0.pt,...,rng_state_rank3.pt,dcp}`
  - `checkpoint-200/{metadata.json,rng_state_rank0.pt,...,rng_state_rank3.pt,dcp}`
  - each DCP directory contains `.metadata` plus `__0_0.distcp` through
    `__3_0.distcp`.
- Validation artifacts:
  - 12 MP4s exist: four each at steps `0`, `100`, and `200`.
  - Downloaded to
    `/tmp/tdm_pilot_sde_200_4e7d9e76_validation_named/`.
  - All 12 decoded with temporary `imageio`/`imageio-ffmpeg` tooling as
    `(77, 448, 832, 3)` `uint8`.
  - All basic non-blank checks passed: pixel std range
    `11.896296..39.797680`; first-vs-last frame absolute mean delta range
    `1.095667..4.911918`.

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

## Completed Pilot Checks

- Job completion: passed.
- Metrics JSONL persisted with 200 loss rows: passed.
- Finite `total_loss`, `fake_score_loss`, and `generator_loss`: passed.
- `fake_score_loss` did not collapse to all zeros: passed.
- `update_student` followed `generator_update_interval=5`: passed.
- Terminal-sigma diagnostics stayed clean: passed.
- Validation MP4s exist at expected steps and decode successfully: passed.
- Video frame count and resolution match the Wan validation config: passed.
- Pixel statistics are not constant/all-black/all-white/corrupt: passed.
- DCP checkpoint directories exist for expected saved steps: passed.

What this still will not prove:

- TDM convergence.
- Flow-matching bridge equivalence to the paper in output quality.
- Whether the distilled model is good.
- Whether 4-step Wan TDM beats or matches DMD or Self-Forcing baselines.

## Current Remaining Work

1. Compare validation artifacts visually and decide whether a
   500-step run or baseline DMD/Self-Forcing comparison is warranted.
2. Decide whether the duplicate final `checkpoint-200` save is acceptable as
   existing trainer behavior or should be cleaned up before PR.
3. Before any PR, re-check issue/PR state, rerun Modal tests, and prepare a
   draft PR only if explicitly requested.
