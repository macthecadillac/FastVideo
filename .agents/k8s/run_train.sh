#!/usr/bin/env bash
# Inside-pod training script for the issue-775 TDM bsz=4 500-step run.
#
# This is exec'd from the workstation driver via `kubectl exec`. It assumes:
#   * The init container already cloned the repo into
#     ${POD_OUTPUT_ROOT}/repo and pre-downloaded the Wan model + dataset.
#   * ${POD_OUTPUT_ROOT} is on the shared PVC so logs and checkpoints
#     survive pod termination.
#   * The pod has 4 GPUs visible via CUDA.
#
# Args:
#   $1 = POD_OUTPUT_ROOT (e.g. /workspace/run/issue-775/tdm_bsz4_500_k8s_xxx)
#
# Side effects:
#   * Writes train.log to ${POD_OUTPUT_ROOT}/logs/train.log
#   * Writes checkpoints to ${POD_OUTPUT_ROOT}/output/checkpoint-*
#   * Writes JSONL tracker rows to ${POD_OUTPUT_ROOT}/output/tracker/metrics.jsonl
#   * Touches ${POD_OUTPUT_ROOT}/output/.train_done on completion (success OR failure)

set -euo pipefail

POD_OUTPUT_ROOT="${1:?Usage: run_train.sh <POD_OUTPUT_ROOT>}"
REPO_DIR="${POD_OUTPUT_ROOT}/repo"
LOG_DIR="${POD_OUTPUT_ROOT}/logs"
OUT_DIR="${POD_OUTPUT_ROOT}/output"
DONE_FLAG="${OUT_DIR}/.train_done"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${POD_OUTPUT_ROOT}/cache/hf"
export HF_HUB_CACHE="${POD_OUTPUT_ROOT}/cache/hf/hub"
export HF_DATASETS_CACHE="${POD_OUTPUT_ROOT}/cache/hf/datasets"
export WANDB_MODE=disabled
export WANDB_API_KEY=""
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

echo "[run_train] start at $(date -u +%FT%TZ)"
echo "[run_train] nvidia-smi:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "[run_train] torchrun torch.distributed availability:"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"

set +e
torchrun --standalone --nproc_per_node=4 \
    -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml \
    --training.data.data_path data/Wan-Syn_77x448x832_600k \
    --training.data.train_batch_size 4 \
    --training.loop.gradient_accumulation_steps 1 \
    --training.loop.max_train_steps 500 \
    --method.generator_update_interval 1 \
    --training.checkpoint.output_dir "${OUT_DIR}" \
    --training.checkpoint.training_state_checkpointing_steps 100 \
    --training.checkpoint.checkpoints_total_limit 6 \
    --training.tracker.project_name distillation_wan \
    --training.tracker.run_name "${RUN_NAME:-tdm_bsz4_500_k8s}" \
    --callbacks.validation.every_steps 0 \
    --pipeline.flow_shift 8 \
    --pipeline.dmd_sample_type ode \
    2>&1 | tee "${LOG_DIR}/train.log"
TRAIN_RC=$?
set -e

echo "[run_train] torchrun exited rc=${TRAIN_RC} at $(date -u +%FT%TZ)"
date -u +%FT%TZ > "${DONE_FLAG}"
echo "rc=${TRAIN_RC}" >> "${DONE_FLAG}"
exit "${TRAIN_RC}"
