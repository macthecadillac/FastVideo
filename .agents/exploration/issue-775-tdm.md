# Issue 775 TDM Handoff

Compacted: 2026-07-01
Last updated: 2026-07-02 during resumed interval-1 run

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
- Latest pushed issue-branch commit before this handoff update:
  `a0a4f93a` `[misc]: record TDM visual review`

## GitHub State

Last checked: 2026-07-02 with `gh` authenticated as `macthecadillac`.

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
- `ea7e8704` `[misc]: record TDM 200-step pilot`
- `13d76da9` `[misc]: record TDM teacher comparison artifact`
- `a0a4f93a` `[misc]: record TDM visual review`

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
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_pilot_sde_200_4e7d9e76_validation_named/`
    after the 2026-07-02 restart wiped `/tmp`.
  - All 12 decoded with temporary `imageio`/`imageio-ffmpeg` tooling as
    `(77, 448, 832, 3)` `uint8`.
  - All basic non-blank checks passed: pixel std range
    `11.896296..39.797680`; first-vs-last frame absolute mean delta range
    `1.095667..4.911918`.

Prompt 0 teacher/student comparison:

- User asked for both the Wan teacher output and TDM student outputs for one
  prompt, plus the prompt text, for visual inspection.
- Chosen prompt index: `0` from
  `examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json`.
- Prompt text:

  ```text
  In the video, a woman is elegantly showcasing her earrings, bringing attention to their intricate design with a gentle touch of her fingers. She is bathed in ambient purple and pink lighting, which casts a soft glow on her delicate features and enhances the vivid tones of her lipstick and eye makeup. Her hair is styled to frame her face smoothly, emphasizing the contours of her jawline and cheekbones. The background features a blurred neon light, adding an artistic and modern touch to the overall aesthetic.
  ```

- Teacher generation Modal app: `ap-wVSQpGeoOvGexEGqkzn4hv`
- Teacher output root in Modal volume:
  `/root/data/issue_775_tdm_prompt0_compare`
- Teacher settings:
  - Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
  - Pipeline: normal `WanPipeline`
  - Inference steps: `50`
  - Seed: `1000`
  - Resolution: `448x832`
  - Frames/FPS: `77` / `16`
  - Guidance scale: `6.0`
  - Flow shift: `8.0`
  - Logged generation time: `216.996s`; e2e latency `219.733s`
  - Peak memory: `14352.989 MB`
- Local assembled comparison folder:
  `/home/toolbox/FastVideo/outputs/issue-775-tdm/prompt0_teacher_student_compare/`
- Files in the comparison folder:
  - `teacher/issue_775_tdm_prompt0_compare/wan_teacher_50_steps_seed1000_prompt0.mp4`
  - `teacher/issue_775_tdm_prompt0_compare/wan_teacher_50_steps_seed1000_prompt0.json`
  - `teacher/issue_775_tdm_prompt0_compare/prompt0.txt`
  - `tdm_student_4step/tdm_student_step000_4steps_seed1000_prompt0.mp4`
  - `tdm_student_4step/tdm_student_step100_4steps_seed1000_prompt0.mp4`
  - `tdm_student_4step/tdm_student_step200_4steps_seed1000_prompt0.mp4`
  - `README.md`
- Note: these are qualitative artifacts. The teacher and TDM student use
  different sampler trajectories, so they are not expected to be pixel-matched.
  Local `ffprobe` failed to initialize the H.264 decoder in this environment,
  but the Modal log confirms the teacher MP4 was generated, decoded, and saved.
- User visual assessment:
  - Teacher output is imperfect but clearly conditioned by the prompt, vivid,
    and detailed.
  - TDM student output does not get visibly worse from `step0` to `step100`
    to `step200`.
  - TDM student output remains extremely blurred at all inspected steps, with
    no identifiable objects; user described it as if the teacher output were
    seen through heavily frosted glass.
- Interpretation:
  - This does not prove numerical failure because an untrained/base Wan model
    sampled with the aggressive 4-step DMD schedule can be very blurry, and the
    200-step pilot contains only about 40 student/generator updates.
  - It does mean the pilot has not shown qualitative learning yet. Before
    spending on a much longer run, first distinguish expected 4-step base blur
    from an integration/validation issue.

Prompt 0 targeted blur diagnostics:

- Base Wan 4-step DMD baseline:
  - First attempt app `ap-AD8N2PjcdD064G53krRQZb` failed before model code
    because Modal could not check out newest handoff-only commit `a0a4f93a`
    (`fatal: reference is not a tree`). Code diagnostics therefore used
    known-visible code commit `ea7e8704`; production code was unchanged between
    those commits.
  - Successful app: `ap-w5GquHcspYVAYi3QadKluL`
  - Commit: `ea7e8704502aaa0485aa9dba921416c66c1542b6`
  - Pipeline: `WanDMDPipeline`
  - Settings: same prompt, seed `1000`, `448x832`, `77` frames, `fps=16`,
    `guidance_scale=6.0`, `embedded_cfg_scale=6.0`, `flow_shift=8.0`,
    `num_inference_steps=4`,
    `dmd_denoising_steps=[1000,750,500,250]`.
  - Modal output:
    `/root/data/issue_775_tdm_prompt0_compare/base_dmd_4step/base_wan_dmd_4steps_seed1000_prompt0.mp4`
  - Local output:
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/prompt0_teacher_student_compare/base_dmd_4step/base_dmd_4step/base_wan_dmd_4steps_seed1000_prompt0.mp4`
  - Visual/stat result: base Wan 4-step DMD is also heavily blurred.
- DCP export / normal inference loader probe:
  - App: `ap-WxB3oDIhzdiRT8PCK7DjRZ`
  - `dcp_to_diffusers` successfully exported the step-200 student transformer
    to
    `/root/data/issue_775_tdm_prompt0_compare/student_step200_diffusers`,
    but normal `VideoGenerator.from_pretrained(export_path, ...)` failed
    because `model_index.json` was a broken symlink. Probe app
    `ap-jyjjWsfqEaM9Q9IZOqbxEx` confirmed `Path.exists() == False` for that
    symlink even though the name appeared in directory listings. Root cause:
    the converter copies HF snapshot files with `symlinks=True`, preserving
    links to blobs outside the export root.
  - A follow-up attempt to load the exported student transformer through
    `init_weights_from_safetensors` failed in app `ap-3gQijLcOOHmKJrx251E0IV`
    with missing normal-loader key
    `blocks.0.attn2.to_k.base_layer.bias`. The exported state still contains
    LoRA-wrapper key structure and is not directly loadable by standard
    inference without a LoRA merge/export path.
- Live training-module high-step validation:
  - Temporary diagnostic script:
    `scripts/diagnostics/tdm_resume_validate.py`, applied to Modal with
    `--apply-local-patch`; this script was only for the diagnostic and is not
    intended to remain in the final branch.
  - App: `ap-Tqz1iA99t1AO6Lwr7ztDpE`
  - Commit: `ea7e8704502aaa0485aa9dba921416c66c1542b6` plus the temporary
    local diagnostic script patch.
  - Command:
    `torchrun --standalone --nproc_per_node=4 scripts/diagnostics/tdm_resume_validate.py --checkpoint /root/data/tdm_pilot_sde_200_4e7d9e76/checkpoint-200 --output-dir /root/data/issue_775_tdm_prompt0_compare/student_step200_live_wan_50step --inference-steps 50`
  - Result: completed successfully on 4x L40S. The checkpoint resumed at step
    `200`, EMA callback initialized/restored, validation offloaded optimizer
    state plus teacher/critic transformers, and the normal `WanPipeline` used
    the already-provided live student transformer for a 50-step sample.
  - Modal output:
    `/root/data/issue_775_tdm_prompt0_compare/student_step200_live_wan_50step`
  - Local output:
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/prompt0_teacher_student_compare/student_step200_live_wan_50step/`
  - Downloaded MP4s:
    `validation/validation_step_200_inference_steps_50_video_0.mp4` through
    `video_3.mp4`. The four copies are expected because the single validation
    prompt was padded across four SP groups; use `video_0` as the canonical
    comparison file.
  - OpenCV metadata for `video_0`: opened successfully, `77` frames, `16`
    fps, `832x448`. First-frame stats: mean `84.0549`, std `88.2849`.
  - Comparison stats from the same OpenCV probe:
    - Teacher 50-step first frame: mean `85.0922`, std `88.6851`
    - TDM step-200 4-step first frame: mean `63.6605`, std `16.8114`
    - Base Wan DMD 4-step first frame: mean `63.4405`, std `16.6299`
  - Representative contact sheet:
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/prompt0_teacher_student_compare/thumbnails/prompt0_frame38_contact_sheet.png`
  - Visual result: teacher 50-step and live step-200 student sampled at 50
    steps are both vivid and prompt-conditioned. Base Wan 4-step DMD and TDM
    step-200 4-step are both blurred/frosted.
- Current interpretation:
  - The step-200 student checkpoint and live validation/EMA application are
    not globally broken; when sampled with normal 50-step Wan, the live
    student still produces a clear prompt-conditioned video.
  - The observed failure is concentrated in the aggressive 4-step DMD/TDM
    trajectory. The 200-step pilot did not visibly improve over the blurred
    base 4-step DMD baseline, which is plausible with only about 40 generator
    updates but is not convergence evidence.
  - Separate cleanup is needed for `dcp_to_diffusers`: copied HF snapshot
    symlinks should be materialized, and exported LoRA-wrapped student weights
    need a merge/export path before normal inference can load them.

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
- Base Wan 4-step DMD baseline for prompt 0 was generated: passed.
- Live step-200 student resumed and sampled with normal 50-step Wan validation:
  passed.
- Live 50-step student output is clear/prompt-conditioned while 4-step base and
  4-step TDM outputs are blurred: passed.

## Active Diagnostic Plan

Next run: interval-1 student-update diagnostic for the same Wan TDM LoRA setup.

- Rationale: the 200-step SDE pilot used `generator_update_interval=5`, so it
  produced only about 40 student/generator updates. The critic was healthy and
  validation/EMA wiring is now proven, but the 4-step student output did not
  improve over the blurred base 4-step DMD baseline.
- First diagnostic to run: `max_train_steps=200` with
  `method.generator_update_interval=1`. This gives about 200 student updates,
  matching the student-update count of a `1000` outer-step run with interval
  `5`, but with fewer critic-only outer steps.
- Keep all other meaningful settings aligned with the current default TDM
  config: SDE rollout, LoRA rank/alpha, Wan 2.1 T2V 1.3B, 4-step validation
  schedule `[1000,750,500,250]`, seed `1000`, `448x832`, `77` frames.
- Expected validation targets: 4-step videos at steps `0`, `100`, and `200`,
  metrics JSONL, and DCP checkpoints. Compare prompt 0 against the existing
  teacher 50-step, base Wan DMD 4-step, and interval-5 200-step TDM artifacts.
- Decision rule:
  - If interval `1` stays stable and improves 4-step contrast/detail, consider
    changing the recommended TDM diagnostic/default cadence or running a
    longer interval-1 pilot.
  - If interval `1` stays stable but remains blurred, prioritize objective /
    sigma schedule / weighting debugging over simply increasing outer steps.
  - If interval `1` is unstable, keep interval `5` and run the longer 1000-step
    convergence pilot as originally planned.

Interval-1 run status:

- First launch app `ap-ZS7yR5ozq2vlVb8ruIOmox` failed before training because
  the command missed the Modal dataset override and used the YAML placeholder
  `data/Wan-Syn_77x448x832_600k`. Root error:
  `FileNotFoundError: No parquet files found under dataset path`.
- Corrected launch app: `ap-vX9fvoypEPLOOXFRYPygPo`
- Commit: `ea7e8704502aaa0485aa9dba921416c66c1542b6` plus CLI-only config
  overrides; later branch commits are handoff-only.
- Output root:
  `/root/data/tdm_pilot_sde_200_interval1_35888898_dataset`
- Corrected command summary:
  - `generator_update_interval=1`
  - `max_train_steps=200`
  - dataset:
    `/root/data/.cache/datasets--wlsaidhi--crush-smol_processed_t2v/snapshots/67dd07f2163ad2b3397f8b3d8125b67ca452dc85/combined_parquet_dataset`
  - checkpoints every `100` steps
  - JSONL tracker only
  - validation every `100` steps
  - validation offloads training state and unloads pipeline after validation
- Status observed in logs:
  - Corrected run reached staged parquet dataset, initialized JSONL tracker,
    instantiated validation and EMA, completed step-0 validation, entered
    training, and logged both `grad_norm/student` and `grad_norm/critic`
    starting at step `1`.
  - Through step `100`, every step logged student and critic metrics,
    confirming interval `1` is active and stable through the first half of the
    run. Approximate steady-state step time: `40` seconds.
  - Step `100` started checkpoint save to
    `/root/data/tdm_pilot_sde_200_interval1_35888898_dataset/checkpoint-100`,
    offloaded optimizer state plus teacher/critic for validation, completed
    4-step validation, restored validation state, and resumed training through
    at least step `102`.
  - The local session was interrupted after logs had reached about step `111`.
    After the interruption, `/tmp` worktrees were wiped and had to be
    recreated. Modal volume listing showed only partial committed artifacts:
    `checkpoint-100`, tracker, and validation MP4s for steps `0` and `100`.
    No `checkpoint-200` or step-200 validation files were present, so the
    interval-1 diagnostic is not complete yet.
- Resume plan:
  - Recreate `/tmp/fastvideo-worktrees/issue-775-tdm` at branch
    `issue/775-tdm` and `/tmp/fastvideo-worktrees/interleavethinker-modal` at
    branch `interleavethinker`.
  - Relaunch the same interval-1 command with
    `training.checkpoint.resume_from_checkpoint=/root/data/tdm_pilot_sde_200_interval1_35888898_dataset/checkpoint-100`,
    same output root, and `max_train_steps=200`.
  - After completion, download tracker plus validation MP4s and compare prompt
    0 against teacher/base-DMD/interval-5 artifacts.
- Resume status:
  - Resume app `ap-XVo1dVzlskj0Bs1UUMNbvH` launched from branch
    `issue/775-tdm` using commit
    `ea7e8704502aaa0485aa9dba921416c66c1542b6`.
  - The resumed job loaded
    `/root/data/tdm_pilot_sde_200_interval1_35888898_dataset/checkpoint-100`
    and logged `Checkpoint loaded; resuming from step=100`.
  - It reran step-100 validation after restore, restored optimizer/teacher/
    critic state, then resumed training past the previous interrupted point.
  - As of 2026-07-02 07:47 UTC, logs reached step `150`; each resumed step
    has logged student and critic gradients plus `update_student`, confirming
    `generator_update_interval=1` remains active.
  - The resumed job later advanced normally through step `192`. At
    `2026-07-02 08:15:07 UTC`, the local stream appeared to stop at step `193`
    after `grad_norm/student`; no later streamed `grad_norm/critic`, loss row,
    EMA row, progress-bar advance, checkpoint, validation, or process-exit logs
    followed for several minutes. Later downloaded JSONL showed this was a
    stream/log artifact: the durable tracker contains complete step-`193`
    rows, so the actual stall happened after step `193` completed and before
    step `194` completed.
  - Independent `uvx modal app logs ap-XVo1dVzlskj0Bs1UUMNbvH` checks matched
    the local stream: stdout stopped at the step-193 student-gradient line and
    stderr showed only progress-bar output through step `192`. `uvx modal app
    list` still showed the app as active with one task, so this appears to be
    an in-step hang/deadlock rather than a completed or failed run.
  - No step-200 checkpoint or validation artifacts are expected from this app
    unless the hang unexpectedly resolves before it is stopped. The last
    committed durable artifacts remain `checkpoint-100`, tracker, and
    validation MP4s for steps `0` and `100`.
  - The app was stopped manually with
    `uvx modal app stop ap-XVo1dVzlskj0Bs1UUMNbvH --yes`; the local launcher
    then exited with Modal `RemoteError` caused by "user stopped from CLI".
    After two stop commands, `uvx modal app list` still showed this app in
    `stopping...` state with one task, so Modal may take additional time to
    tear down the deadlocked container.
  - Post-stop Modal volume listing confirmed only these durable artifacts:
    `checkpoint-100`, `tracker`, and the eight validation videos for steps
    `0` and `100`. There is still no `checkpoint-200` or step-200 validation.
  - Persisted artifacts were downloaded to
    `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_pilot_sde_200_interval1_35888898_dataset/`.
    This local directory contains `tracker/{config.json,metrics.jsonl,
    artifacts.jsonl,files/run.yaml}` plus eight MP4s.
  - Tracker summary:
    - raw JSONL rows: `840`
    - unique steps: `1..193`
    - duplicate metric groups: steps `101..117`, all four row types; this
      matches overlap between the original interrupted job and the resumed
      job.
    - de-duplicated rows: `772`
    - de-duplicated loss/student-grad/critic-grad/EMA rows: `193` each
    - `update_student` true on all de-duplicated steps `1..193`
    - nonfinite scalar metrics: `0`
    - de-duplicated `total_loss`: min `0.1882624775`, max `0.9230759740`,
      mean `0.4580486829`, first `0.3315637112`, last `0.4681196511`
    - de-duplicated `fake_score_loss`: min `0.0004032278`, max
      `0.0016866034`, mean `0.0010733638`, first `0.0015967983`, last
      `0.0014704503`
    - de-duplicated `generator_loss`: min `0.1868882924`, max
      `0.9219567180`, mean `0.4569753191`, first `0.3299669027`, last
      `0.4666492045`
  - Downloaded video sanity check with `ffprobe`: all eight MP4s report
    `832x448`, `77` frames, duration `4.812500`.

What this still will not prove:

- TDM convergence.
- Flow-matching bridge equivalence to the paper in output quality.
- Whether the distilled model is good.
- Whether 4-step Wan TDM beats or matches DMD or Self-Forcing baselines.

## Current Remaining Work

1. Investigate the interval-1 hang before running a longer interval-1 pilot.
   The hang occurred after logging step-193 student grad and before the
   step-193 critic/loss/EMA rows, so the most likely debugging surface is the
   critic-side backward/optimizer/distributed synchronization path rather than
   data loading or validation.
   - 2026-07-02 continuation: code inspection narrowed the exact sequence.
     `grad_norm/*` rows are produced by `GradNormClipCallback` after method
     backward and before optimizer stepping. The JSONL tracker writes and
     flushes the row before logging `JSONL tracker step=...`, so the persisted
     step-193 `grad_norm/student` row means student clipping and tracker write
     completed. The next operation is critic gradient clipping.
   - Added gated callback instrumentation in
     `fastvideo/train/callbacks/grad_clip.py`: `debug_log` plus
     `debug_log_steps`. With default values off, training behavior is
     unchanged. When enabled, every rank logs begin/end for each role's grad
     clip, local parameter/gradient counts, gradient tensor types/dtypes/
     devices, elapsed time, and returned norm.
   - Local non-test checks for this instrumentation: `python3 -m py_compile
     fastvideo/train/callbacks/grad_clip.py`, `git diff --check`,
     `uvx pre-commit run --files fastvideo/train/callbacks/grad_clip.py`
     (yapf reformatted; mypy hook failed before checking due to the known
     invalid worktree package name `issue-775-tdm`), and direct
     `uvx mypy --explicit-package-bases fastvideo/train/callbacks/grad_clip.py`
     passed.
   - Next Modal diagnostic: run from `checkpoint-100` with
     `generator_update_interval=1`, validation/checkpoint saving disabled,
     and `callbacks.grad_clip.debug_log=true` focused around steps `188..195`.
     If it hangs again, the last debug line should identify whether all ranks
     entered critic clipping and which phase failed to return.
   - First attempted debug Modal app `ap-Km1HANsi5SYrfaretqn4h7` was stopped
     during startup before training because the initial instrumentation used
     the default FastVideo logger behavior, which only logs from local rank 0.
     Patched the debug logger call to pass `local_main_process_only=False` so
     all ranks emit grad-clip begin/end lines. The stop produced the expected
     Modal `RemoteError`/SIGINT logs while the T5 encoder was loading; no
     training-step evidence came from that app.
   - Active all-rank Modal diagnostic: app `ap-og6ZwCuU1gbTgog6T08ub2`,
     commit `3550048bc446d837fc20e948471615720df22c85`, output root
     `/root/data/tdm_debug_interval1_clip_allranks_195_3550048b`.
     It resumes from
     `/root/data/tdm_pilot_sde_200_interval1_35888898_dataset/checkpoint-100`,
     disables validation/checkpoint saves, runs to step 195, and logs
     grad-clip begin/end from all ranks for steps 188-195.
   - Result: the no-validation/no-checkpoint control completed through step
     `195` and committed Modal volume output. Persisted tracker:
     `/home/toolbox/FastVideo/outputs/issue-775-tdm/tdm_debug_interval1_clip_allranks_195_3550048b/metrics.jsonl`.
     Summary: `380` JSONL rows; steps `101..195` (`95` unique steps);
     `95` student-grad rows, `95` critic-grad rows, `95` loss rows, `95` EMA
     rows; nonfinite scalar metrics `0`. All ranks reported `DTensor` grads
     for both student and critic in the debug window. Critic grad clipping
     returned on every rank at steps `188..195`; step `193` also logged
     critic, loss, and EMA rows. Therefore the prior interval-1 stop is not a
     deterministic grad-clip deadlock in the validation-free training loop.
   - Important correction: the downloaded tracker for the earlier stopped
     interval-1 resume contains complete step-`193` rows (`grad_norm/student`,
     `grad_norm/critic`, loss metrics, and `ema/decay`). The streamed logs had
     appeared to stop after `grad_norm/student`, but the durable JSONL shows
     step `193` completed. The actual stalled interval was after step `193`
     completed and before step `194` completed.
2. Decide the next training diagnostic budget. The current evidence supports
   running a longer pilot only as a 4-step-convergence check, not to debug
   checkpoint loading or EMA application, but the interval-1 hang must be
   addressed or bounded first.
3. If running longer, keep the same prompt 0 comparison workflow:
   teacher 50-step, base Wan DMD 4-step, TDM student 4-step at checkpoints, and
   optional live-student 50-step spot checks. The expected success signal is
   that TDM 4-step starts gaining contrast/detail and prompt-conditioned
   objects relative to the blurred base 4-step DMD baseline.
4. Fix or defer `dcp_to_diffusers` export limitations:
   materialize copied HF symlinks and add a LoRA merge/export path so normal
   inference can load a DCP-exported TDM student.
5. Decide whether the duplicate final `checkpoint-200` save is acceptable as
   existing trainer behavior or should be cleaned up before PR.
6. Before any PR, re-check issue/PR state, rerun Modal tests, and prepare a
   draft PR only if explicitly requested.
