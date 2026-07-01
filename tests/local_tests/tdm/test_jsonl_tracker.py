# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path

from fastvideo.training.trackers import initialize_trackers


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_jsonl_tracker_writes_metrics_artifacts_and_files(tmp_path: Path) -> None:
    source_file = tmp_path / "run.yaml"
    source_file.write_text("training: {}\n", encoding="utf-8")

    tracker = initialize_trackers(
        ["jsonl"],
        experiment_name="unused",
        config={"tensor_like": [1, 2, 3]},
        log_dir=str(tmp_path / "tracker"),
        run_name="smoke",
    )

    tracker.log(
        {
            "loss": 1.25,
            "nonfinite": math.nan,
        },
        step=3,
    )
    video = tracker.video(
        str(tmp_path / "sample.mp4"),
        caption="prompt",
        fps=16,
    )
    tracker.log_artifacts(
        {"validation": [video]},
        step=3,
    )
    tracker.log_file(
        str(source_file),
        name="run.yaml",
    )

    metric_rows = _read_jsonl(tmp_path / "tracker" / "metrics.jsonl")
    assert metric_rows[0]["step"] == 3
    metrics = metric_rows[0]["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["loss"] == 1.25
    assert metrics["nonfinite"] == {
        "_type": "nonfinite_float",
        "value": "nan",
    }

    artifact_rows = _read_jsonl(tmp_path / "tracker" / "artifacts.jsonl")
    assert artifact_rows[0]["step"] == 3
    assert artifact_rows[1]["file"] == {
        "name": "run.yaml",
        "path": str(tmp_path / "tracker" / "files" / "run.yaml"),
        "source": str(source_file),
    }
    assert (tmp_path / "tracker" / "config.json").is_file()
    saved_run = tmp_path / "tracker" / "files" / "run.yaml"
    assert saved_run.read_text(encoding="utf-8") == "training: {}\n"
