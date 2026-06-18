# SPDX-License-Identifier: Apache-2.0
"""Prompt refinement helpers for PromptRL-style training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

RefinementMode = Literal["none", "dataset_column", "template", "model"]


@dataclass(frozen=True, slots=True)
class PromptRefinementConfig:
    """YAML-backed prompt refinement settings."""

    enabled: bool = False
    mode: RefinementMode = "none"
    refined_prompt_key: str = "refined_prompt"
    template: str = "{prompt}"
    num_original_prompts: int = 0

    @classmethod
    def from_mapping(cls, raw: Any) -> PromptRefinementConfig:
        if raw is None:
            return cls()
        if isinstance(raw, bool):
            return cls(enabled=raw, mode="dataset_column" if raw else "none")
        if not isinstance(raw, dict):
            raise ValueError(f"method.prompt_refinement must be a bool or mapping, got {type(raw).__name__}")
        supported_keys = {
            "enabled",
            "mode",
            "num_original_prompts",
            "refined_prompt_key",
            "template",
        }
        unsupported_keys = sorted(set(raw) - supported_keys)
        if unsupported_keys:
            raise ValueError(f"Unsupported method.prompt_refinement key(s): {unsupported_keys}")

        enabled = _coerce_bool(raw.get("enabled", True))
        mode = str(raw.get("mode", "dataset_column" if enabled else "none") or "none").strip().lower()
        if mode not in {"none", "dataset_column", "template", "model"}:
            raise ValueError("method.prompt_refinement.mode must be one of "
                             "{none, dataset_column, template, model}")
        refined_prompt_key = str(raw.get("refined_prompt_key", "refined_prompt") or "refined_prompt")
        template = str(raw.get("template", "{prompt}") or "{prompt}")
        num_original_prompts = int(raw.get("num_original_prompts", 0) or 0)
        if num_original_prompts < 0:
            raise ValueError("method.prompt_refinement.num_original_prompts must be non-negative")
        if mode == "template" and "{prompt}" not in template:
            raise ValueError("method.prompt_refinement.template must contain {prompt}")
        if not enabled:
            mode = "none"
        return cls(
            enabled=enabled,
            mode=mode,  # type: ignore[arg-type]
            refined_prompt_key=refined_prompt_key,
            template=template,
            num_original_prompts=num_original_prompts,
        )


@dataclass(frozen=True, slots=True)
class PromptRefinementResult:
    original_prompts: list[str]
    prompts: list[str]
    refined_mask: list[bool]
    policy_mask: list[bool] = field(default_factory=list)
    refiner_log_probs: torch.Tensor | None = None
    old_refiner_log_probs: torch.Tensor | None = None
    metadata: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected = len(self.prompts)
        if len(self.original_prompts) != expected:
            raise ValueError("PromptRefinementResult original_prompts/prompts length mismatch")
        if len(self.refined_mask) != expected:
            raise ValueError("PromptRefinementResult refined_mask length mismatch")
        if not self.policy_mask:
            object.__setattr__(self, "policy_mask", [False for _ in range(expected)])
        elif len(self.policy_mask) != expected:
            raise ValueError("PromptRefinementResult policy_mask length mismatch")
        if not self.metadata:
            object.__setattr__(self, "metadata", [{} for _ in range(expected)])
        elif len(self.metadata) != expected:
            raise ValueError("PromptRefinementResult metadata length mismatch")
        _validate_optional_tensor(self.refiner_log_probs, expected, "refiner_log_probs")
        _validate_optional_tensor(self.old_refiner_log_probs, expected, "old_refiner_log_probs")

    @property
    def refined_count(self) -> int:
        return sum(1 for refined in self.refined_mask if refined)


def refine_prompt_batch(
    raw_batch: dict[str, Any],
    config: PromptRefinementConfig,
    *,
    prompt_refiner: Any | None = None,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, Any], PromptRefinementResult]:
    """Return a prompt-refined shallow copy of ``raw_batch``."""
    original_prompts = _extract_prompts(raw_batch)
    if not config.enabled or config.mode == "none":
        return dict(raw_batch), PromptRefinementResult(
            original_prompts=original_prompts,
            prompts=list(original_prompts),
            refined_mask=[False for _ in original_prompts],
            policy_mask=[False for _ in original_prompts],
        )

    if config.mode == "model":
        result = _refine_with_model(
            raw_batch,
            original_prompts,
            config,
            prompt_refiner=prompt_refiner,
            generator=generator,
        )
        return _replace_prompts(raw_batch, result.prompts), result

    candidates = _candidate_refined_prompts(raw_batch, config)
    refined_prompts: list[str] = []
    refined_mask: list[bool] = []
    for idx, prompt in enumerate(original_prompts):
        if idx < config.num_original_prompts:
            refined_prompts.append(prompt)
            refined_mask.append(False)
            continue

        refined = prompt
        if config.mode == "dataset_column":
            candidate = candidates[idx] if idx < len(candidates) else ""
            refined = candidate or prompt
        elif config.mode == "template":
            refined = config.template.format(prompt=prompt)

        refined_prompts.append(refined)
        refined_mask.append(refined != prompt)

    return _replace_prompts(raw_batch, refined_prompts), PromptRefinementResult(
        original_prompts=original_prompts,
        prompts=refined_prompts,
        refined_mask=refined_mask,
        policy_mask=[False for _ in refined_prompts],
    )


def _refine_with_model(
    raw_batch: dict[str, Any],
    original_prompts: list[str],
    config: PromptRefinementConfig,
    *,
    prompt_refiner: Any | None,
    generator: torch.Generator | None,
) -> PromptRefinementResult:
    if prompt_refiner is None:
        raise ValueError("method.prompt_refinement.mode='model' requires a prompt_refiner role")
    refine_fn = getattr(prompt_refiner, "refine_prompts", None)
    if not callable(refine_fn):
        raise ValueError("prompt_refiner must implement refine_prompts(...) for PromptRL model refinement")

    refinement_indices = list(range(min(config.num_original_prompts, len(original_prompts)), len(original_prompts)))
    if not refinement_indices:
        return PromptRefinementResult(
            original_prompts=original_prompts,
            prompts=list(original_prompts),
            refined_mask=[False for _ in original_prompts],
            policy_mask=[False for _ in original_prompts],
        )

    prompts_to_refine = [original_prompts[idx] for idx in refinement_indices]
    model_output = refine_fn(
        prompts=prompts_to_refine,
        raw_batch=raw_batch,
        config=config,
        generator=generator,
    )
    model_result = _coerce_model_refinement_output(model_output, prompts_to_refine)

    refined_prompts = list(original_prompts)
    refined_mask = [False for _ in original_prompts]
    policy_mask = [False for _ in original_prompts]
    metadata: list[Any] = [{} for _ in original_prompts]
    for offset, prompt_idx in enumerate(refinement_indices):
        refined_prompt = model_result.prompts[offset]
        refined_prompts[prompt_idx] = refined_prompt
        refined_mask[prompt_idx] = refined_prompt != original_prompts[prompt_idx]
        policy_mask[prompt_idx] = True
        metadata[prompt_idx] = model_result.metadata[offset]

    return PromptRefinementResult(
        original_prompts=original_prompts,
        prompts=refined_prompts,
        refined_mask=refined_mask,
        policy_mask=policy_mask,
        refiner_log_probs=_scatter_optional_tensor(
            model_result.refiner_log_probs,
            refinement_indices,
            len(original_prompts),
        ),
        old_refiner_log_probs=_scatter_optional_tensor(
            model_result.old_refiner_log_probs,
            refinement_indices,
            len(original_prompts),
        ),
        metadata=metadata,
    )


def _coerce_model_refinement_output(
    output: Any,
    original_prompts: list[str],
) -> PromptRefinementResult:
    expected = len(original_prompts)
    if isinstance(output, PromptRefinementResult):
        if len(output.prompts) != expected:
            raise ValueError("prompt_refiner returned a PromptRefinementResult with the wrong prompt count")
        return output

    if isinstance(output, dict):
        prompts_raw = output.get("prompts", output.get("refined_prompts"))
        if prompts_raw is None:
            raise ValueError("prompt_refiner output mapping must include 'prompts' or 'refined_prompts'")
        prompts = _coerce_prompt_list(prompts_raw, expected)
        return PromptRefinementResult(
            original_prompts=list(original_prompts),
            prompts=prompts,
            refined_mask=[new != old for old, new in zip(original_prompts, prompts, strict=True)],
            policy_mask=[True for _ in prompts],
            refiner_log_probs=_coerce_optional_tensor(output.get("log_probs"), expected, "log_probs"),
            old_refiner_log_probs=_coerce_optional_tensor(
                output.get("old_log_probs"),
                expected,
                "old_log_probs",
            ),
            metadata=_coerce_metadata(output.get("metadata"), expected),
        )

    if isinstance(output, tuple):
        if len(output) not in {2, 3}:
            raise ValueError("prompt_refiner tuple output must be (prompts, log_probs[, metadata])")
        prompts = _coerce_prompt_list(output[0], expected)
        metadata = _coerce_metadata(output[2], expected) if len(output) == 3 else [{} for _ in prompts]
        return PromptRefinementResult(
            original_prompts=list(original_prompts),
            prompts=prompts,
            refined_mask=[new != old for old, new in zip(original_prompts, prompts, strict=True)],
            policy_mask=[True for _ in prompts],
            refiner_log_probs=_coerce_optional_tensor(output[1], expected, "log_probs"),
            metadata=metadata,
        )

    prompts = _coerce_prompt_list(output, expected)
    return PromptRefinementResult(
        original_prompts=list(original_prompts),
        prompts=prompts,
        refined_mask=[new != old for old, new in zip(original_prompts, prompts, strict=True)],
        policy_mask=[True for _ in prompts],
    )


def _coerce_prompt_list(value: Any, expected: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("prompt_refiner prompts must be returned as a list")
    if len(value) != expected:
        raise ValueError(f"prompt_refiner returned {len(value)} prompts, expected {expected}")
    return [str(prompt) for prompt in value]


def _coerce_metadata(value: Any, expected: int) -> list[Any]:
    if value is None:
        return [{} for _ in range(expected)]
    if isinstance(value, list):
        if len(value) != expected:
            raise ValueError(f"prompt_refiner returned {len(value)} metadata records, expected {expected}")
        return list(value)
    return [value for _ in range(expected)]


def _coerce_optional_tensor(
    value: Any,
    expected: int,
    name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
    _validate_optional_tensor(tensor, expected, name)
    return tensor


def _validate_optional_tensor(
    value: torch.Tensor | None,
    expected: int,
    name: str,
) -> None:
    if value is None:
        return
    if value.ndim == 0 or int(value.shape[0]) != expected:
        raise ValueError(f"PromptRefinementResult {name} must have batch dimension {expected}, "
                         f"got {tuple(value.shape)}")


def _scatter_optional_tensor(
    values: torch.Tensor | None,
    indices: list[int],
    total: int,
) -> torch.Tensor | None:
    if values is None:
        return None
    output = torch.zeros(
        (total, ) + tuple(values.shape[1:]),
        device=values.device,
        dtype=values.dtype,
    )
    index_tensor = torch.tensor(indices, device=values.device, dtype=torch.long)
    output[index_tensor] = values
    return output


def _extract_prompts(raw_batch: dict[str, Any]) -> list[str]:
    infos = raw_batch.get("info_list")
    if isinstance(infos, list) and infos:
        prompts: list[str] = []
        for info in infos:
            if isinstance(info, dict):
                prompts.append(str(info.get("prompt") or info.get("caption") or ""))
            else:
                prompts.append("")
        return prompts
    captions = raw_batch.get("caption_text")
    if isinstance(captions, list):
        return [str(caption) for caption in captions]
    raise ValueError("Could not find prompts in batch info_list or caption_text")


def _candidate_refined_prompts(
    raw_batch: dict[str, Any],
    config: PromptRefinementConfig,
) -> list[str]:
    top_level = raw_batch.get(config.refined_prompt_key)
    if isinstance(top_level, list):
        return [str(prompt or "") for prompt in top_level]

    infos = raw_batch.get("info_list")
    if isinstance(infos, list):
        candidates = []
        for info in infos:
            if isinstance(info, dict):
                candidates.append(str(info.get(config.refined_prompt_key) or ""))
            else:
                candidates.append("")
        return candidates
    return []


def _replace_prompts(
    raw_batch: dict[str, Any],
    prompts: list[str],
) -> dict[str, Any]:
    out = dict(raw_batch)
    infos = raw_batch.get("info_list")
    if isinstance(infos, list) and infos:
        copied_infos = []
        for idx, info in enumerate(infos):
            if isinstance(info, dict):
                copied = dict(info)
                target_key = "prompt" if "prompt" in copied else "caption"
                copied[target_key] = prompts[idx]
                copied_infos.append(copied)
            else:
                copied_infos.append(info)
        out["info_list"] = copied_infos

    captions = raw_batch.get("caption_text")
    if isinstance(captions, list):
        out["caption_text"] = list(prompts)
    return out


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError("method.prompt_refinement.enabled must be a boolean")
