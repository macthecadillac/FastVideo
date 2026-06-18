# SPDX-License-Identifier: Apache-2.0
"""Prompt refinement helpers for PromptRL-style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RefinementMode = Literal["none", "dataset_column", "template"]


@dataclass(frozen=True, slots=True)
class PromptRefinementConfig:
    """YAML-backed prompt refinement settings."""

    enabled: bool = False
    mode: RefinementMode = "none"
    refined_prompt_key: str = "refined_prompt"
    template: str = "{prompt}"
    num_original_prompts: int = 0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> PromptRefinementConfig:
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
        if mode not in {"none", "dataset_column", "template"}:
            raise ValueError("method.prompt_refinement.mode must be one of "
                             "{none, dataset_column, template}")
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

    @property
    def refined_count(self) -> int:
        return sum(1 for refined in self.refined_mask if refined)


def refine_prompt_batch(
    raw_batch: dict[str, Any],
    config: PromptRefinementConfig,
) -> tuple[dict[str, Any], PromptRefinementResult]:
    """Return a prompt-refined shallow copy of ``raw_batch``."""
    original_prompts = _extract_prompts(raw_batch)
    if not config.enabled or config.mode == "none":
        return dict(raw_batch), PromptRefinementResult(
            original_prompts=original_prompts,
            prompts=list(original_prompts),
            refined_mask=[False for _ in original_prompts],
        )

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
    )


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
