#!/usr/bin/env python3
"""Validate durable outputs before issue-775 inference runs."""

import json
import math
import sys
from pathlib import Path


def iter_numbers(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_numbers(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield value


def main() -> None:
    output_dir = Path(sys.argv[1])
    done_text = (output_dir / ".train_done").read_text(encoding="utf-8")
    if "rc=0" not in done_text.splitlines():
        raise SystemExit(f"training did not complete successfully:\n{done_text}")

    for step in (100, 200, 300, 400, 500):
        checkpoint = output_dir / f"checkpoint-{step}"
        metadata_path = checkpoint / "metadata.json"
        if not metadata_path.is_file():
            raise SystemExit(f"missing checkpoint metadata: {metadata_path}")
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
            if any(not math.isfinite(float(value)) for value in iter_numbers(row)):
                nonfinite.append(line_number)

    missing_steps = sorted(set(range(1, 501)) - steps)
    if missing_steps:
        raise SystemExit(f"tracker is missing training steps: {missing_steps[:20]}")
    if nonfinite:
        raise SystemExit(f"tracker has nonfinite values on lines: {nonfinite[:20]}")

    print(
        "TRAINING_OUTPUT_OK",
        f"rows={row_count}",
        f"unique_steps={len(steps)}",
        "checkpoints=100,200,300,400,500",
    )


if __name__ == "__main__":
    main()
