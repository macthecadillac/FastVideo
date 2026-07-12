#!/usr/bin/env bash
# Four-GPU compatibility and two-step global-batch-4 training preflight.

set -euo pipefail

POD_OUTPUT_ROOT="${1:?Usage: run_preflight.sh <POD_OUTPUT_ROOT>}"
REPO_DIR="${POD_OUTPUT_ROOT}/repo"
PREFLIGHT_DIR="${POD_OUTPUT_ROOT}/preflight"
LOG_DIR="${POD_OUTPUT_ROOT}/logs"
SHARED_HF_HOME="${SHARED_HF_HOME:-/workspace/run/issue-775/shared-cache/hf}"
mkdir -p "${PREFLIGHT_DIR}" "${LOG_DIR}"

[[ -f "${POD_OUTPUT_ROOT}/.stage_done" ]]
[[ -L "${REPO_DIR}/data/Wan-Syn_77x448x832_600k" ]]
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_CACHE="${SHARED_HF_HOME}/hub"
export HF_DATASETS_CACHE="${SHARED_HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_API_KEY=""
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

echo "[preflight] start $(date -u +%FT%TZ) arch=$(uname -m)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
[[ "${GPU_COUNT}" == "4" ]]

cat > /tmp/issue775_nccl_preflight.py <<'PYDIST'
import os

import torch
import torch.distributed as dist


dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
assert world_size == 4, world_size
value = torch.tensor([float(rank)], device=f"cuda:{os.environ['LOCAL_RANK']}")
dist.all_reduce(value)
assert value.item() == 6.0, value.item()
if rank == 0:
    print("NCCL_PREFLIGHT_OK world_size=4 reduced=6.0")
dist.destroy_process_group()
PYDIST
torchrun --standalone --nproc_per_node=4 /tmp/issue775_nccl_preflight.py

echo "[preflight] starting two-step global-batch-4 TDM smoke"
torchrun --standalone --nproc_per_node=4 \
    -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml \
    --training.data.data_path data/Wan-Syn_77x448x832_600k \
    --training.data.train_batch_size 1 \
    --training.loop.gradient_accumulation_steps 1 \
    --training.loop.max_train_steps 2 \
    --method.generator_update_interval 1 \
    --training.checkpoint.output_dir "${PREFLIGHT_DIR}" \
    --training.checkpoint.training_state_checkpointing_steps 0 \
    --training.checkpoint.checkpoints_total_limit 0 \
    --training.tracker.trackers "[jsonl]" \
    --training.tracker.project_name distillation_wan \
    --training.tracker.run_name "${RUN_NAME:-tdm_bsz4_500_k8s}_preflight" \
    --callbacks.validation.every_steps 0 \
    --pipeline.flow_shift 8 \
    --pipeline.dmd_sample_type ode \
    2>&1 | tee "${LOG_DIR}/preflight.log"

python3 - "${PREFLIGHT_DIR}" <<'PYCONFIG'
import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
config_path = root / "tracker" / "config.json"
metrics_path = root / "tracker" / "metrics.jsonl"
config = json.loads(config_path.read_text(encoding="utf-8"))
training = config["training"]
distributed = training["distributed"]
local_batch = int(training["data"]["train_batch_size"])
gradient_accumulation = int(training["loop"]["gradient_accumulation_steps"])
world_size = int(distributed["num_gpus"])
sp_size = int(distributed["sp_size"])
if world_size % sp_size:
    raise SystemExit(f"world_size={world_size} is not divisible by sp_size={sp_size}")
data_parallel_groups = world_size // sp_size
global_batch = local_batch * data_parallel_groups * gradient_accumulation
if (local_batch, global_batch) != (1, 4):
    raise SystemExit(
        f"invalid batch dimensions: local={local_batch} global={global_batch}"
    )
if training["tracker"]["trackers"] != ["jsonl"]:
    raise SystemExit(f"JSONL tracker is not selected: {training['tracker']['trackers']}")
steps = {
    int(row["step"])
    for line in metrics_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
    for row in [json.loads(line)]
}
if steps != {1, 2}:
    raise SystemExit(f"preflight tracker steps are incomplete: {sorted(steps)}")
if list(root.glob("checkpoint-*")):
    raise SystemExit("preflight unexpectedly wrote training checkpoints")
print(
    "PREFLIGHT_CONFIG_OK",
    f"local_batch={local_batch}",
    f"global_batch={global_batch}",
    "tracker=jsonl",
    f"steps={sorted(steps)}",
)
PYCONFIG

touch "${PREFLIGHT_DIR}/.preflight_done"
echo "[preflight] complete $(date -u +%FT%TZ)"
