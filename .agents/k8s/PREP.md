# Issue 775 TDM - corrected K8s batch-4 run plan

Updated: 2026-07-12
Status: implemented and awaiting separate launch approval

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

## Staging and preflight

`run_k8s.sh` first server-side dry-runs both manifests. It then runs a
CPU-only staging pod that:

1. Verifies the FastVideo image starts on ARM64 and imports Torch.
2. Verifies UID/GID 1000 can write the Lustre run directory.
3. Idempotently checks out the exact branch commit from the public fork.
4. Downloads `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` into the shared cache.
5. Downloads `FastVideo/Wan-Syn_77x448x832_600k` into the shared cache.
6. Uses the exact path returned by `snapshot_download` for the training-data
   symlink under `repo/data/` and verifies a parquet file is present.
7. Writes `.stage_done` only after every check succeeds.

Only after staging succeeds does the driver create the four-GB200 pod. Before
long training, `run_preflight.sh` requires exactly four CUDA devices, runs a
four-rank NCCL all-reduce, and executes two real TDM steps with local batch
size 1 on each of four data-parallel groups. It then checks the JSONL tracker's
resolved run configuration and requires local batch 1, global batch 4, steps
1 and 2, and no preflight checkpoints. Any failure leaves the pod available
for inspection and prevents the 500-step process from starting.

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
code. Every new-format checkpoint writes `.complete` only after DCP and RNG
state barriers finish. `resume_k8s.sh` deletes retained terminal pod objects,
recreates the GPU pod from the durable manifest, and resumes from the newest
checkpoint carrying that marker. It refuses inference unless `rc=0`,
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
RUN_NAME=tdm-bsz4-500-k8s-<timestamp> \
    bash .agents/k8s/run_k8s.sh
```

Reconnect, validate, infer, and download:

```bash
KUBECONFIG=/home/sandbox/.kube/config \
RUN_NAME=<same-run-name> \
    bash .agents/k8s/resume_k8s.sh
```

Artifacts are copied to `outputs/issue-775-tdm/k8s/${RUN_NAME}/`. The GPU pod
is left running only for artifact verification and should then be deleted.
