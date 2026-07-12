from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from fastvideo.training.trackers import _sanitize_jsonl_value


_VALIDATOR_PATH = Path(__file__).with_name("validate_output.py")
_SPEC = importlib.util.spec_from_file_location("issue775_validate_output", _VALIDATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VALIDATOR)


def _write_valid_output(output_dir: Path) -> None:
    output_dir.mkdir()
    (output_dir / ".train_done").write_text("rc=0\n", encoding="utf-8")
    for step in (100, 200, 300, 400, 500):
        checkpoint = output_dir / f"checkpoint-{step}"
        dcp_dir = checkpoint / "dcp"
        dcp_dir.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(
            json.dumps({"step": step, "completion_marker": ".complete"}),
            encoding="utf-8",
        )
        (checkpoint / ".complete").write_text("complete\n", encoding="utf-8")
        (dcp_dir / ".metadata").write_text("dcp metadata\n", encoding="utf-8")

    tracker_dir = output_dir / "tracker"
    tracker_dir.mkdir()
    with (tracker_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for step in range(1, 501):
            handle.write(json.dumps({"step": step, "metrics": {"loss": 1.0}}))
            handle.write("\n")


def test_rejects_jsonl_nonfinite_float_marker(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_valid_output(output_dir)
    marker = _sanitize_jsonl_value(float("nan"))
    assert marker == {"_type": "nonfinite_float", "value": "nan"}
    with (output_dir / "tracker" / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 500, "metrics": {"loss": marker}}))
        handle.write("\n")

    with pytest.raises(SystemExit, match="nonfinite_float"):
        _VALIDATOR.validate_output(output_dir)
