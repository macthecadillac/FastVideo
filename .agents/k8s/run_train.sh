#!/usr/bin/env bash
# Run the 500-step issue-775 TDM training workload inside the GB200 pod.

set -euo pipefail

POD_OUTPUT_ROOT="${1:?Usage: run_train.sh <POD_OUTPUT_ROOT>}"
REPO_DIR="${POD_OUTPUT_ROOT}/repo"
LOG_DIR="${POD_OUTPUT_ROOT}/logs"
OUT_DIR="${POD_OUTPUT_ROOT}/output"
DONE_FLAG="${OUT_DIR}/.train_done"
SHARED_HF_HOME="${SHARED_HF_HOME:-/workspace/run/issue-775/shared-cache/hf}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"
rm -f "${DONE_FLAG}" "${DONE_FLAG}.tmp"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_CACHE="${SHARED_HF_HOME}/hub"
export HF_DATASETS_CACHE="${SHARED_HF_HOME}/datasets"
export WANDB_MODE=disabled
export WANDB_API_KEY=""
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

TRAIN_ARGS=(
    --config examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml
    --training.data.data_path data/Wan-Syn_77x448x832_600k
    --training.data.train_batch_size 1
    --training.loop.gradient_accumulation_steps 1
    --training.loop.max_train_steps 500
    --method.generator_update_interval 1
    --training.checkpoint.output_dir "${OUT_DIR}"
    --training.checkpoint.training_state_checkpointing_steps 100
    --training.checkpoint.checkpoints_total_limit 6
    --training.tracker.trackers "[jsonl]"
    --training.tracker.project_name distillation_wan
    --training.tracker.run_name "${RUN_NAME:-tdm_bsz4_500_k8s}"
    --callbacks.validation.every_steps 0
    --pipeline.flow_shift 8
    --pipeline.dmd_sample_type ode
)
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    TRAIN_ARGS+=(--training.checkpoint.resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

echo "[run_train] start $(date -u +%FT%TZ)"
echo "[run_train] output=${OUT_DIR} resume=${RESUME_FROM_CHECKPOINT:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"

set +e
torchrun --standalone --nproc_per_node=4 \
    -m fastvideo.train.entrypoint.train \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/train.log"
TRAIN_RC=$?
set -e

{
    echo "completed_at=$(date -u +%FT%TZ)"
    echo "rc=${TRAIN_RC}"
} > "${DONE_FLAG}.tmp"
mv "${DONE_FLAG}.tmp" "${DONE_FLAG}"
echo "[run_train] torchrun exited rc=${TRAIN_RC} at $(date -u +%FT%TZ)"
exit "${TRAIN_RC}"
