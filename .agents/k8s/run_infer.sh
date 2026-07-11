#!/usr/bin/env bash
# Inside-pod post-training inference script for the issue-775 TDM bsz=4 500-step run.
#
# This is exec'd from the workstation driver via `kubectl exec` after training
# completes. It generates TDM 4-step student videos from the trained
# checkpoint-500 using the TDM validation callback, then also generates
# teacher/base Wan 50-step videos for the same four prompts for an
# apples-to-apples visual comparison.
#
# Args:
#   $1 = POD_OUTPUT_ROOT (e.g. /workspace/run/issue-775/tdm_bsz4_500_k8s_xxx)
#
# Outputs:
#   ${POD_OUTPUT_ROOT}/validation_step500/student_tdm_4step/validation_step_500_inference_steps_4_video_{0..3}.mp4
#   ${POD_OUTPUT_ROOT}/validation_step500/teacher_wan_50step/teacher_wan_50step_prompt{0..3}.mp4
#   ${POD_OUTPUT_ROOT}/validation_step500/teacher_wan_50step/prompts.json

set -euo pipefail

POD_OUTPUT_ROOT="${1:?Usage: run_infer.sh <POD_OUTPUT_ROOT>}"
REPO_DIR="${POD_OUTPUT_ROOT}/repo"
LOG_DIR="${POD_OUTPUT_ROOT}/logs"
OUT_DIR="${POD_OUTPUT_ROOT}/output"
VALIDATION_DIR="${POD_OUTPUT_ROOT}/validation_step500"

mkdir -p "${LOG_DIR}" "${VALIDATION_DIR}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${POD_OUTPUT_ROOT}/cache/hf"
export HF_HUB_CACHE="${POD_OUTPUT_ROOT}/cache/hf/hub"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

CHECKPOINT_DIR="${OUT_DIR}/checkpoint-500"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: ${CHECKPOINT_DIR} not found. Was training run to step 500?" >&2
    exit 1
fi

STUDENT_OUT="${VALIDATION_DIR}/student_tdm_4step"
TEACHER_OUT="${VALIDATION_DIR}/teacher_wan_50step"
mkdir -p "${STUDENT_OUT}" "${TEACHER_OUT}"

cp -f examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json \
    "${VALIDATION_DIR}/validation_4.json"

echo "[run_infer] start at $(date -u +%FT%TZ)"
echo "[run_infer] checkpoint: ${CHECKPOINT_DIR}"

set +e
torchrun --standalone --nproc_per_node=1 \
    -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml \
    --training.data.data_path data/Wan-Syn_77x448x832_600k \
    --training.data.train_batch_size 1 \
    --training.loop.gradient_accumulation_steps 1 \
    --training.loop.max_train_steps 0 \
    --training.checkpoint.output_dir "${OUT_DIR}" \
    --training.checkpoint.resume_from_checkpoint "${CHECKPOINT_DIR}" \
    --training.tracker.project_name distillation_wan \
    --training.tracker.run_name "${RUN_NAME:-tdm_bsz4_500_k8s}_validation" \
    --callbacks.validation.every_steps 1 \
    --callbacks.validation.sampling_steps "[4]" \
    --callbacks.validation.sampling_timesteps "[1000,750,500,250]" \
    --callbacks.validation.guidance_scale 6.0 \
    --callbacks.validation.output_dir "${STUDENT_OUT}" \
    --pipeline.flow_shift 8 \
    --pipeline.dmd_sample_type ode \
    2>&1 | tee "${LOG_DIR}/student_infer.log"
STUDENT_RC=$?
set -e
echo "[run_infer] student 4-step rc=${STUDENT_RC} at $(date -u +%FT%TZ)"

cat > /tmp/run_teacher_validation.py <<'PY'
"""Standalone teacher/base Wan 50-step validation for the four TDM prompts."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/root/.cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastvideo import VideoGenerator

VALIDATION_JSON = sys.argv[1]
OUTPUT_DIR = Path(sys.argv[2])

with open(VALIDATION_JSON, "r") as f:
    data = json.load(f)["data"]
prompts = [d["caption"] for d in data]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / "prompts.json", "w") as f:
    json.dump(
        [{"index": i, "prompt": p} for i, p in enumerate(prompts)],
        f, indent=2,
    )

gen = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,
)
for i, prompt in enumerate(prompts):
    print(f"=== prompt {i} ===")
    out = gen.generate_video(
        prompt=prompt,
        output_path=str(OUTPUT_DIR / f"teacher_wan_50step_prompt{i}.mp4"),
        height=448,
        width=832,
        num_frames=77,
        num_inference_steps=50,
        guidance_scale=6.0,
        flow_shift=8.0,
        fps=16,
        seed=1000,
    )
    print(f"  saved {out}")
PY
set +e
python /tmp/run_teacher_validation.py \
    "${VALIDATION_DIR}/validation_4.json" \
    "${TEACHER_OUT}" \
    2>&1 | tee "${LOG_DIR}/teacher_infer.log"
TEACHER_RC=$?
set -e
echo "[run_infer] teacher 50-step rc=${TEACHER_RC} at $(date -u +%FT%TZ)"

echo "[run_infer] ffprobe sanity check"
for f in "${STUDENT_OUT}"/*.mp4 "${TEACHER_OUT}"/*.mp4; do
    [[ -f "$f" ]] || continue
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,r_frame_rate,nb_frames \
        -of default=noprint_wrappers=1 "$f" || true
done

echo "[run_infer] done at $(date -u +%FT%TZ)"
