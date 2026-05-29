# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage, StageVerificationError
from fastvideo.pipelines.stages.validators import (
    StageValidators as V,
    VerificationResult,
    assert_tensor_has_no_nan,
    tensor_validation_context,
)


class _EchoTensorStage(PipelineStage):

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        return result

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        return batch


def _fastvideo_args(enable_full_tensor_validation: bool) -> FastVideoArgs:
    args = FastVideoArgs(model_path="test-model")
    args.enable_stage_verification = True
    args.enable_full_tensor_validation = enable_full_tensor_validation
    return args


def test_tensor_validators_skip_nan_scans_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTVIDEO_FULL_TENSOR_VALIDATION", raising=False)
    value = torch.tensor([1.0, float("nan")])

    assert V.is_tensor(value)
    assert V.tensor_with_dims(value, 1)
    assert V.none_or_tensor(value)
    assert V.list_of_tensors([value])
    assert_tensor_has_no_nan(value, "value", enabled=False)


def test_tensor_validators_reject_nans_when_full_validation_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTVIDEO_FULL_TENSOR_VALIDATION", raising=False)
    value = torch.tensor([1.0, float("nan")])

    with tensor_validation_context(True):
        assert not V.is_tensor(value)
        result = VerificationResult()
        result.add_check("latents", value, V.is_tensor)

        assert not result.is_valid()
        assert "tensor contains 1 NaN values" in result.get_failure_summary()

    with pytest.raises(AssertionError, match="value contains nan"):
        assert_tensor_has_no_nan(value, "value", enabled=True)


def test_pipeline_stage_structural_verification_allows_nans_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTVIDEO_FULL_TENSOR_VALIDATION", raising=False)
    batch = ForwardBatch(data_type="test", latents=torch.tensor([1.0, float("nan")]))

    result = _EchoTensorStage()(batch, _fastvideo_args(enable_full_tensor_validation=False))

    assert result is batch


def test_pipeline_stage_full_tensor_validation_rejects_nans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTVIDEO_FULL_TENSOR_VALIDATION", raising=False)
    batch = ForwardBatch(data_type="test", latents=torch.tensor([1.0, float("nan")]))

    with pytest.raises(StageVerificationError, match="tensor contains 1 NaN values"):
        _EchoTensorStage()(batch, _fastvideo_args(enable_full_tensor_validation=True))
