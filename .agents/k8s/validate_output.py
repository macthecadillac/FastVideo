#!/usr/bin/env python3
"""Validate durable outputs before issue-775 inference runs."""

import json
import math
import sys
from pathlib import Path


EXPECTED_CHECKPOINTS = (50, 100, 200)


def iter_nonfinite_values(value, path="$"):
    if isinstance(value, dict):
        if value.get("_type") == "nonfinite_float":
            yield f"{path}=nonfinite_float({value.get('value')})"
            return
        for key, child in value.items():
            yield from iter_nonfinite_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_nonfinite_values(child, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            yield f"{path}={value}"


def validate_output(output_dir: Path) -> None:
    done_text = (output_dir / ".train_done").read_text(encoding="utf-8")
    if "rc=0" not in done_text.splitlines():
        raise SystemExit(f"training did not complete successfully:\n{done_text}")

    for step in EXPECTED_CHECKPOINTS:
        checkpoint = output_dir / f"checkpoint-{step}"
        metadata_path = checkpoint / "metadata.json"
        marker_path = checkpoint / ".complete"
        dcp_metadata_path = checkpoint / "dcp" / ".metadata"
        if not metadata_path.is_file():
            raise SystemExit(f"missing checkpoint metadata: {metadata_path}")
        if not marker_path.is_file():
            raise SystemExit(f"checkpoint save did not complete: {checkpoint}")
        if not dcp_metadata_path.is_file():
            raise SystemExit(f"missing DCP metadata: {dcp_metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("step", -1)) != step:
            raise SystemExit(f"invalid checkpoint step in {metadata_path}: {metadata.get('step')}")

    metrics_path = output_dir / "tracker" / "metrics.jsonl"
    if not metrics_path.is_file():
        raise SystemExit(f"missing tracker metrics: {metrics_path}")

    steps = set()
    row_count = 0
    nonfinite = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_count += 1
            if isinstance(row.get("step"), (int, float)):
                steps.add(int(row["step"]))
            markers = list(iter_nonfinite_values(row))
            if markers:
                nonfinite.append((line_number, markers))

    missing_steps = sorted(set(range(1, EXPECTED_CHECKPOINTS[-1] + 1)) - steps)
    if missing_steps:
        raise SystemExit(f"tracker is missing training steps: {missing_steps[:20]}")
    if nonfinite:
        raise SystemExit(f"tracker has nonfinite values: {nonfinite[:20]}")

    print(
        "TRAINING_OUTPUT_OK",
        f"rows={row_count}",
        f"unique_steps={len(steps)}",
        f"checkpoints={','.join(str(step) for step in EXPECTED_CHECKPOINTS)}",
    )


def main() -> None:
    validate_output(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
