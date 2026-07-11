# Issue 775 TDM - K8s bsz=4 500-step run prep

Started: 2026-07-11
Stage: pre-launch prep (no run executed yet)

## Goal

A 500-step TDM training run with `train_batch_size=4` on the
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` student using
`data/Wan-Syn_77x448x832_600k`, followed by a 4-step student visual
generation and a 50-step base Wan teacher generation for the same
four prompts. The whole pipeline runs on a Kubernetes cluster
(`oci-bth-aiaccelerator-prd005`, namespace `vllm`, 4×GB200 node),
and artifacts are pulled back to the workstation.

## Disconnected-run plan

The user will not be connected for the long training run. The
launch-time driver therefore:

- Applies the pod and waits for the init container to begin.
- Polls the HF cache size every 20s for up to 90s during the init
  container to estimate the dataset download rate and report a
  projected total init time. This is the only "stay around" phase.
- After the init container finishes, exec's the training script
  inside the pod under `nohup ... &` and writes the training PID
  to `${POD_OUTPUT_ROOT}/output/train.pid`. The training process
  is detached from the exec'd shell and survives the local session
  ending.
- Prints a clear "you can disconnect" banner with the pod name,
  the run name, the training PID, and the resume command.
- Exits.

A separate `resume_k8s.sh` script is for the reconnect session
(maybe hours/days later). It is idempotent:

- If the pod is gone (e.g. K8s evicted it), recreates it from the
  saved launch manifest. The HF cache and the staged repo live on
  the PVC, so the recreated init container is a no-op for the
  dataset/model and only re-checks-out the repo commit.
- Polls the training done flag (or the training process PID)
  until training finishes.
- Runs the inference script (4-step student + 50-step teacher).
- Pulls `output/` and `validation_step500/` to the workstation.
- Leaves the pod Running so the user can re-run inference with
  different settings if needed.

## Why K8s and not Modal/DGX

- DGX Spark: a previous 500-step run already produced degraded 4-step
  videos (per user visual review at the end of the prior handoff).
  Trying a longer/larger-batch run is the next reasonable experiment,
  but the cost is real (DGX bsz=4 500-step would be ~67-125h).
- Modal: the project rule says Modal is for unit tests, not long
  training runs.
- This K8s cluster: 4×GB200 per node with the FastVideo dev image
  available, the dataset PVC mountable, and a public-ish HPC cluster
  the user is authed to. Step time on GB200 Blackwell should be a
  meaningful fraction of the DGX H100 step time, so the wall-clock
  cost of "more training" is much lower.

## Cluster access verified

- The `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest`
  image is **public** (confirmed by anonymous GET against the OCI
  image index endpoint). No `imagePullSecrets` is needed. The
  cluster's `image-preload` daemonset also pulls a `ghcr.io` image
  without any pull secret, consistent with that.
- The K8s user is `vlm-mal004` in namespace `vllm`. The `gh-secret`,
  `github-token-secret`, and `hf-token` secrets exist.
- The kubeconfig's `proxy-url: socks5://localhost:1080` does not work
  from this machine. The workstation must rewrite it to
  `socks5://host.containers.internal:1080` (kubectl does not accept
  the `socks5h` scheme).
- 19+ schedulable GB200 nodes (4× NVIDIA GB200, ARM64, ~1 TiB RAM
  per node). The default `lustre-pvc-vllm` (124 TiB, RWX) is bound
  and used by other vLLM workloads in the namespace.

## Files prepared in this worktree

Under `.agents/k8s/`:

- `pod.yaml`: Pod manifest with an init container that clones the
  FastVideo repo at the pinned commit, snapshots the Wan 2.1 T2V
  1.3B model + the Wan-Syn dataset into the PVC, and a long-lived
  main container. No `imagePullSecrets`. 4× GB200, 96 CPU,
  768 GiB RAM.
- `run_k8s.sh`: Workstation launch driver. Renders the manifest,
  applies the pod, polls briefly for a download-rate estimate,
  exec's the training script under `nohup`, prints a
  "you can disconnect" banner, and exits.
- `resume_k8s.sh`: Workstation reconnect driver. Verifies the
  pod (recreating it from the saved manifest if evicted), waits
  for training to finish, runs the inference script, and pulls
  artifacts to the workstation. Idempotent and safe to re-run.
- `run_train.sh`: Inside-pod training. 4-GPU `torchrun`, TDM config
  with `train_batch_size=4`, `gradient_accumulation_steps=1`,
  `max_train_steps=500`, `generator_update_interval=1`, validation
  disabled.
- `run_infer.sh`: Inside-pod inference. Reuses the TDM training
  entrypoint with `every_steps=1` and `max_train_steps=0` to force
  one validation pass from checkpoint-500, then runs a standalone
  `VideoGenerator` script for the 50-step base Wan teacher.
- `_envsubst.py`: Whitelisted envsubst for the pod manifest so that
  runtime shell variables (e.g. `${GH_TOKEN}`) are not expanded
  early.
- `PREP.md`: human-readable summary of the run plan.

## Launch plan (not yet executed)

1. From the worktree `/tmp/fastvideo-worktrees/issue-775-tdm-k8s`:
   `KUBECONFIG=/home/toolbox/.kube/config K8S_NAMESPACE=vllm \
        RUN_NAME=tdm_bsz4_500_k8s_<timestamp> \
        bash .agents/k8s/run_k8s.sh`
2. The driver applies the pod, waits for the init container to
   start, then polls the HF cache size every 20s for up to 90s.
   The driver reports the download rate and an estimated init
   completion time.
3. The driver waits for the init container to finish (clones the
   repo at `a990855eb`, snapshots Wan 2.1 T2V 1.3B + Wan-Syn into
   the PVC).
4. The driver exec's the training script inside the pod under
   `nohup ... &`, writes the training PID to a file, and prints a
   clear banner saying "you can disconnect".
5. The driver exits. Training continues on the cluster.
6. Later, the user reconnects and runs:
   `KUBECONFIG=/home/toolbox/.kube/config RUN_NAME=<same> \
        bash .agents/k8s/resume_k8s.sh`
7. The resume driver waits for training to finish, runs the
   inference script, and pulls artifacts to
   `outputs/issue-775-tdm/k8s/${RUN_NAME}/output/` and
   `outputs/issue-775-tdm/k8s/${RUN_NAME}/validation/`.
8. The pod is left `Running` for any follow-up. Teardown is manual
   with `kubectl -n vllm delete pod ${POD_NAME}`.

## Expected cost and runtime

- 500 steps at bsz=4. The previous DGX bsz=4 2-step smoke took
  ~488s/step and ~446s/step steady state, so a 500-step run would
  be ~62-67h on the H100 DGX Spark. On GB200 Blackwell the same
  shape is expected to be roughly 1/3 to 1/2 that, but this is
  not yet measured.
- The init container will snapshot the Wan-Syn_77x448x832_600k
  dataset (~1.6 TB on DGX). On the K8s cluster the dataset is
  staged onto the lustre PVC. The first run will pay this cost
  once; subsequent runs can reuse the staged cache.
- The Wan model snapshot is ~30 GB; cheap.

## Known unknowns

- The handoff for issue #775 has a final code commit of
  `5c2a8b9b` and a final handoff commit of `a990855eb`. The K8s
  driver checks out the latter by default. The user may want to
  re-train from an even more recent commit if they want; the env
  var `COMMIT` is overridable.
- The post-training `run_infer.sh` reuses the training entrypoint
  with `every_steps=1` to force one validation pass. This worked on
  DGX. If it does not work on K8s, fallback is a standalone
  `VideoGenerator` script.
- The training command sets `training.data.data_path` to
  `data/Wan-Syn_77x448x832_600k` (relative to the working dir). The
  driver exec's the training script from `${POD_OUTPUT_ROOT}/repo`
  and the dataset is symlinked into
  `${POD_OUTPUT_ROOT}/data/Wan-Syn_77x448x832_600k`, so the path
  needs to be resolved relative to the workdir. The init container
  handles this.
- Pod eviction is the biggest disconnect-related risk. The pod has
  `restartPolicy: Never` and uses the default priority class. If
  K8s evicts the pod, the resume driver will recreate it from the
  saved manifest. Training state is on the PVC, so we do not lose
  training progress.

## Next

- Wait for user confirmation on the prep.
- After user approves, run the launch driver from
  `/tmp/fastvideo-worktrees/issue-775-tdm-k8s`.
