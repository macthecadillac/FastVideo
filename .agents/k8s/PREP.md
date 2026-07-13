# Issue 775 TDM - corrected K8s batch-4 run plan

Updated: 2026-07-13
Status: implemented; each launch still requires explicit approval

## Goal

Train the issue-775 Wan TDM student for 500 steps at global batch size 4,
then compare three student checkpoints against the base Wan model at four
steps and the base Wan teacher at 50 steps. No Kubernetes workload should be
created until the user separately approves launch.

## Cluster layout

- Cluster: `oci-bth-aiaccelerator-prd005`
- Namespace: `vllm`
- Staging: ARM64 `VM.Standard.A2.Flex`, no GPU
- Training/inference: one `BM.GPU.GB200.4` node with four GB200 GPUs
- Durable storage: `lustre-pvc-vllm`, mounted at `/workspace/run`
- Shared HF cache: `/workspace/run/issue-775/shared-cache/hf`
- Per-run root: `/workspace/run/issue-775/${RUN_NAME}`

The pod security context uses UID/GID 1000 but deliberately omits `fsGroup`.
The Lustre root is already group 1000 and group-writable; setting `fsGroup`
caused kubelet to attempt recursive ownership changes across the 124 TiB
filesystem.

## Resources

CPU-only staging pod:

- CPU request/limit: 16/32
- Memory request/limit: 64/120 GiB
- Ephemeral-storage request/limit: 200/400 GiB
- GPU: none

Training pod:

- GPU request/limit: 4/4
- CPU request/limit: 96/144
- Memory request/limit: 768/896 GiB
- Ephemeral-storage request/limit: 200/400 GiB
- `/dev/shm`: 64 GiB
- `/tmp`: 200 GiB

The training pod is created with a scheduling gate, so these requests do not
enter scheduler accounting until staging has completed successfully.

## Staging and preflight

`run_k8s.sh` validates all launch resources with client- and server-side
dry-runs. It then runs a CPU-only staging pod that:

1. Verifies the FastVideo image starts on ARM64 and imports Torch.
2. Verifies UID/GID 1000 can write the Lustre run directory.
3. Requires an explicitly approved full commit SHA, then idempotently checks
   out that exact branch commit from the public fork.
4. Downloads `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` into the shared cache.
5. Downloads `FastVideo/Wan-Syn_77x448x832_600k` into the shared cache.
6. Uses the exact path returned by `snapshot_download` for the training-data
   symlink under `repo/data/` and verifies a parquet file is present.
7. Writes `.stage_done` only after every check succeeds.

Both model and dataset downloads are cache-resuming and guarded by a no-write
watchdog. If a downloader makes no filesystem writes for 20 minutes, staging
terminates that attempt and retries against the same persistent cache, up to
four attempts.

A Kubernetes Job watches the atomic `.stage_done` marker. Its service account
can only read and patch the exact gated GPU pod and read and delete the exact
staging pod. It cannot create arbitrary namespace pods. The supervisor image is
pinned by digest. Only after the marker
matches the approved commit does the Job remove the GPU pod's scheduling gate.
The Job, gated pod, and rendered recovery manifest are stored in the cluster,
so workstation disconnects, restarts, and local `/tmp` cleanup cannot interrupt
the transition. The GPU pod runs `run_preflight.sh` itself before starting
training. Any failure prevents the 500-step process from starting.

## Training parameters

- Config: `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`
- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- Dataset: `data/Wan-Syn_77x448x832_600k`
- GPUs: 4; HSDP shard dimension 4; TP/SP 1
- Local batch size: 1 per data-parallel group
- Global batch size: 4; gradient accumulation: 1
- Steps: 500
- Generator update interval: 1
- TDM timesteps: `[1000, 750, 500, 250]`
- Student and DMD sampling: ODE
- Flow shift: 8
- Student optimizer: LR `2e-6`, betas `[0.0, 0.999]`, weight decay `0.01`
- Critic optimizer: LR `8e-6`, betas `[0.0, 0.999]`
- Gradient clipping: 1.0
- EMA: decay 0.98 from step 0
- Checkpoints: every 100 steps; retain up to 6
- Validation during training: disabled
- W&B: disabled; durable JSONL tracker selected explicitly

The training wrapper writes `.train_done` atomically with the torchrun return
code. The GPU pod runs preflight and training itself, records its shell PID,
and automatically resumes from the newest completed checkpoint when recreated.
Every new-format checkpoint writes `.complete` only after DCP and RNG state
barriers finish. `resume_k8s.sh` can restore a lost local GPU manifest from the
launch ConfigMap before recreating the pod. While staging or its supervisor is
active, the helper exits without creating or releasing a GPU pod. A recreated
post-staging pod has its restored scheduling gate removed explicitly. If a
detached training process dies while its sleeping pod remains Running, set
`RECOVER_DEAD_TRAINING=1` to opt
into one recovery attempt from the newest completed checkpoint; the default is
to stop for inspection. It refuses inference unless `rc=0`,
checkpoints 100 through 500 have matching metadata, DCP metadata, and
completion markers, tracker steps 1 through 500 exist, and tracker values do
not contain numeric nonfinite values or JSONL nonfinite markers. Loss before
the first completed checkpoint remains unrecoverable.

## Comparison outputs

For the same four prompts, seed 1000, guidance 6.0, flow shift 8, 832x448,
77 frames, and 16 fps:

- TDM student, four steps: checkpoints 100, 300, and 500
- Base Wan, four steps
- Base Wan teacher, 50 steps

The inference wrapper requires 20 MP4 files in total and validates resolution,
frame count, and frame rate with `ffprobe`. Any student, base, teacher, or media
validation failure returns nonzero and does not write `.infer_done`. Student
validation uses an isolated runtime output directory with checkpoint saving and
tracking disabled, so it cannot alter retained training checkpoints or metrics.

## Launch and reconnect commands

Launch only after explicit approval:

```bash
KUBECONFIG=/home/sandbox/.kube/config \
K8S_NAMESPACE=vllm \
COMMIT=<approved-full-40-character-commit-sha> \
RUN_NAME=tdm-bsz4-500-k8s-<timestamp> \
    bash .agents/k8s/run_k8s.sh
```

The command returns after submitting staging and its supervisor. For a
confirmed staging-process stall, rerun the same command and run name with
`RESTART_STAGING=1`; the existing pod is replaced while the shared Lustre
cache is retained:

```bash
RESTART_STAGING=1 COMMIT=<approved-full-40-character-commit-sha> \
RUN_NAME=<same-run-name> bash .agents/k8s/run_k8s.sh
```

Reconnect, validate, infer, and download:

```bash
KUBECONFIG=/home/sandbox/.kube/config \
RUN_NAME=<same-run-name> \
    bash .agents/k8s/resume_k8s.sh
```

Artifacts are copied to `outputs/issue-775-tdm/k8s/${RUN_NAME}/`. The GPU pod
is left running only for artifact verification and should then be deleted.
