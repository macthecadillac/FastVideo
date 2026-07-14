#!/usr/bin/env bash
# Generate checkpoint progression plus matched base/teacher comparisons.

set -euo pipefail

POD_OUTPUT_ROOT="${1:?Usage: run_infer.sh <POD_OUTPUT_ROOT>}"
REPO_DIR="${POD_OUTPUT_ROOT}/repo"
LOG_DIR="${POD_OUTPUT_ROOT}/logs"
OUT_DIR="${POD_OUTPUT_ROOT}/output"
VALIDATION_DIR="${POD_OUTPUT_ROOT}/validation_step200"
SHARED_HF_HOME="${SHARED_HF_HOME:-/workspace/run/issue-775/shared-cache/hf}"
INFER_DONE="${VALIDATION_DIR}/.infer_done"
STUDENT_ROOT="${VALIDATION_DIR}/student_tdm_4step"
BASE_OUT="${VALIDATION_DIR}/base_wan_4step"
TEACHER_OUT="${VALIDATION_DIR}/teacher_wan_50step"
mkdir -p "${LOG_DIR}" "${VALIDATION_DIR}"
rm -f "${INFER_DONE}"
rm -rf -- "${STUDENT_ROOT}" "${BASE_OUT}" "${TEACHER_OUT}"

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

python3 .agents/k8s/validate_output.py "${OUT_DIR}"
cp -f examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json \
    "${VALIDATION_DIR}/validation_4.json"

echo "[run_infer] start $(date -u +%FT%TZ)"
for step in 50 100 200; do
    CHECKPOINT_DIR="${OUT_DIR}/checkpoint-${step}"
    STUDENT_OUT="${STUDENT_ROOT}/checkpoint-${step}"
    RUNTIME_OUT="${STUDENT_OUT}/runtime"
    mkdir -p "${STUDENT_OUT}" "${RUNTIME_OUT}"
    echo "[run_infer] student checkpoint-${step}, four steps"
    torchrun --standalone --nproc_per_node=4 \
        -m fastvideo.train.entrypoint.train \
        --config examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml \
        --training.data.data_path data/Wan-Syn_77x448x832_600k \
        --training.data.train_batch_size 1 \
        --training.loop.gradient_accumulation_steps 1 \
        --training.loop.max_train_steps 0 \
        --training.checkpoint.output_dir "${RUNTIME_OUT}" \
        --training.checkpoint.resume_from_checkpoint "${CHECKPOINT_DIR}" \
        --training.checkpoint.training_state_checkpointing_steps 0 \
        --training.checkpoint.checkpoints_total_limit 0 \
        --training.tracker.trackers "[none]" \
        --callbacks.validation.every_steps 1 \
        --callbacks.validation.sampling_steps "[4]" \
        --callbacks.validation.sampling_timesteps "[1000,750,500,250]" \
        --callbacks.validation.guidance_scale 6.0 \
        --callbacks.validation.output_dir "${STUDENT_OUT}" \
        --pipeline.flow_shift 8 \
        --pipeline.dmd_sample_type ode \
        2>&1 | tee "${LOG_DIR}/student_infer_step${step}.log"
done
python3 .agents/k8s/validate_output.py "${OUT_DIR}"

mkdir -p "${BASE_OUT}" "${TEACHER_OUT}"
cat > /tmp/issue775_reference_infer.py <<'PYREF'
import json
import sys
from pathlib import Path

from datasets import load_dataset
from fastvideo import VideoGenerator

validation_json = Path(sys.argv[1])
base_output = Path(sys.argv[2])
teacher_output = Path(sys.argv[3])
data = load_dataset("json", data_files=str(validation_json), split="train", field="data")
prompts = [item["caption"] for item in data]
base_output.mkdir(parents=True, exist_ok=True)
teacher_output.mkdir(parents=True, exist_ok=True)
(teacher_output / "prompts.json").write_text(
    json.dumps([{"index": i, "prompt": prompt} for i, prompt in enumerate(prompts)], indent=2),
    encoding="utf-8",
)


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
        flow_shift=8.0,
    )
    for index, prompt in enumerate(prompts):
        common = {
            "prompt": prompt,
            "height": 448,
            "width": 832,
            "num_frames": 77,
            "guidance_scale": 6.0,
            "fps": 16,
            "seed": 1000,
        }
        generator.generate_video(
            **common,
            num_inference_steps=4,
            output_path=str(base_output / f"base_wan_4step_prompt{index}.mp4"),
        )
        generator.generate_video(
            **common,
            num_inference_steps=50,
            output_path=str(teacher_output / f"teacher_wan_50step_prompt{index}.mp4"),
        )


if __name__ == "__main__":
    main()
PYREF
python3 /tmp/issue775_reference_infer.py \
    "${VALIDATION_DIR}/validation_4.json" \
    "${BASE_OUT}" \
    "${TEACHER_OUT}" \
    2>&1 | tee "${LOG_DIR}/reference_infer.log"

python3 - "${VALIDATION_DIR}" <<'PYVERIFY'
import sys
from fractions import Fraction
from pathlib import Path

import av

root = Path(sys.argv[1])
files = sorted(root.rglob("*.mp4"))
if len(files) != 20:
    raise SystemExit(f"expected 20 MP4 files, found {len(files)}")
for path in files:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        actual = (
            stream.codec_context.width,
            stream.codec_context.height,
            sum(1 for _ in container.decode(video=0)),
            Fraction(stream.average_rate),
        )
    finally:
        container.close()
    expected = (832, 448, 77, Fraction(16, 1))
    if actual != expected:
        raise SystemExit(f"invalid video metadata for {path}: {actual}")
print(f"INFERENCE_OUTPUT_OK videos={len(files)}")
PYVERIFY

touch "${INFER_DONE}"
echo "[run_infer] complete $(date -u +%FT%TZ)"
