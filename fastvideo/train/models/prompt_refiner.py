# SPDX-License-Identifier: Apache-2.0
"""Prompt-refiner model roles for PromptRL-style methods."""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.module_state import apply_trainable

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import TrainingConfig

DEFAULT_PROMPT_REFINEMENT_TEMPLATE = (
    "Rewrite the following text-to-video prompt so it is more specific, "
    "visually grounded, and concise. Return only the rewritten prompt.\n\n"
    "Original prompt: {prompt}\n"
    "Rewritten prompt:")


class CausalLMPromptRefiner(ModelBase):
    """HF causal-LM prompt refiner for PromptRL.

    The role is intentionally independent from diffusion model wrappers. RL
    methods use ``refine_prompts`` during rollout and ``compute_log_probs``
    during prompt-policy optimization.
    """

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        prompt_template: str = DEFAULT_PROMPT_REFINEMENT_TEMPLATE,
        max_new_tokens: int = 96,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        torch_dtype: str | None = "bfloat16",
        trust_remote_code: bool = True,
        lora: LoraConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(trainable=trainable, lora=lora)
        if "{prompt}" not in prompt_template:
            raise ValueError("prompt_refiner.prompt_template must contain {prompt}")
        self._init_from = str(init_from)
        self.prompt_template = str(prompt_template)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.do_sample = bool(do_sample)
        if self.max_new_tokens <= 0:
            raise ValueError("prompt_refiner.max_new_tokens must be positive")
        if self.temperature < 0.0:
            raise ValueError("prompt_refiner.temperature must be non-negative")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError("prompt_refiner.top_p must be in (0, 1]")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._init_from,
            trust_remote_code=bool(trust_remote_code),
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"trust_remote_code": bool(trust_remote_code)}
        dtype = _resolve_torch_dtype(torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.transformer = AutoModelForCausalLM.from_pretrained(self._init_from, **model_kwargs)
        if not self._enable_lora_if_configured(self.transformer):
            self.transformer = apply_trainable(self.transformer, trainable=self._trainable)
        self.transformer.to(device=self.device)
        self.noise_scheduler = None
        self.training_config = training_config

    @torch.no_grad()
    def refine_prompts(
        self,
        *,
        prompts: list[str],
        raw_batch: dict[str, Any],
        config: Any,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        del raw_batch, config
        if not prompts:
            return {
                "prompts": [],
                "log_probs": torch.empty(0, device=self._model_device()),
                "metadata": [],
            }

        input_texts = [self._format_input(prompt) for prompt in prompts]
        encoded = self._tokenize(input_texts)
        generate_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample and self.temperature > 0.0,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
            if generator is not None:
                generate_kwargs["generator"] = generator

        generated_ids = self.transformer.generate(**generate_kwargs)
        prefix_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        refined_prompts = []
        for row, prefix_len, fallback in zip(generated_ids, prefix_lengths, prompts, strict=True):
            response_ids = row[int(prefix_len):]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            refined_prompts.append(response or fallback)

        metadata = [{"input_text": text} for text in input_texts]
        log_probs = self.compute_log_probs(
            original_prompts=prompts,
            refined_prompts=refined_prompts,
            metadata=metadata,
        )
        return {
            "prompts": refined_prompts,
            "log_probs": log_probs.detach(),
            "metadata": metadata,
        }

    def compute_log_probs(
        self,
        *,
        original_prompts: list[str],
        refined_prompts: list[str],
        metadata: list[Any] | None = None,
    ) -> torch.Tensor:
        if len(original_prompts) != len(refined_prompts):
            raise ValueError("original_prompts and refined_prompts must have the same length")
        if not original_prompts:
            return torch.empty(0, device=self._model_device())

        metadata = metadata or [{} for _ in original_prompts]
        input_texts = [
            str(meta.get("input_text")) if isinstance(meta, dict) and meta.get("input_text") else
            self._format_input(prompt)
            for prompt, meta in zip(original_prompts, metadata, strict=True)
        ]
        full_texts = [input_text + refined for input_text, refined in zip(input_texts, refined_prompts, strict=True)]

        prefix = self._tokenize(input_texts)
        full = self._tokenize(full_texts)
        prefix_lengths = prefix["attention_mask"].sum(dim=1)
        outputs = self.transformer(
            input_ids=full["input_ids"],
            attention_mask=full["attention_mask"],
        )
        logits = outputs.logits[:, :-1].float()
        labels = full["input_ids"][:, 1:]
        token_log_probs = torch.log_softmax(logits, dim=-1)
        gathered = torch.gather(token_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        token_positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0) + 1
        response_mask = token_positions >= prefix_lengths.unsqueeze(1)
        response_mask = response_mask & full["attention_mask"][:, 1:].bool()
        return (gathered * response_mask.to(dtype=gathered.dtype)).sum(dim=1)

    def _format_input(self, prompt: str) -> str:
        return self.prompt_template.format(prompt=str(prompt))

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        )
        device = self._model_device()
        return {key: value.to(device=device) for key, value in encoded.items()}

    def _model_device(self) -> torch.device:
        try:
            return next(self.transformer.parameters()).device
        except StopIteration:
            return self.device

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        del raw_batch, generator, latents_source
        raise NotImplementedError("CausalLMPromptRefiner does not prepare diffusion batches")

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del clean_latents, noise, timestep
        raise NotImplementedError("CausalLMPromptRefiner does not implement diffusion noising")

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        del noisy_latents, timestep, batch, conditional, cfg_uncond, attn_kind
        raise NotImplementedError("CausalLMPromptRefiner does not predict diffusion noise")

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        del ctx, grad_accum_rounds
        loss.backward()


def _resolve_torch_dtype(value: str | None) -> torch.dtype | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"", "none", "auto"}:
        return None
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported prompt_refiner.torch_dtype: {value!r}")
