import math
from types import SimpleNamespace

import torch

from fastvideo.train.models.prompt_refiner import (
    CausalLMPromptRefiner,
    _resolve_torch_dtype,
)


class _CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def __call__(self, texts, *, padding, return_tensors):
        del return_tensors
        rows = [[2 + (ord(ch) % 8) for ch in text] for text in texts]
        max_len = max(len(row) for row in rows)
        if padding:
            input_ids = [row + [self.pad_token_id] * (max_len - len(row)) for row in rows]
            attention_mask = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]
        else:
            input_ids = rows
            attention_mask = [[1] * len(row) for row in rows]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class _UniformLM(torch.nn.Module):

    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = int(vocab_size)

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        logits = input_ids.new_zeros((*input_ids.shape, self.vocab_size), dtype=torch.float32)
        return SimpleNamespace(logits=logits + self.weight * 0.0)


def test_causal_lm_prompt_refiner_log_probs_mask_only_refined_response_tokens():
    refiner = object.__new__(CausalLMPromptRefiner)
    refiner.prompt_template = "Prompt: {prompt}\nRewrite:"
    refiner.tokenizer = _CharTokenizer()
    refiner.transformer = _UniformLM(vocab_size=16)

    log_probs = refiner.compute_log_probs(
        original_prompts=["a cat"],
        refined_prompts=["sharp cat"],
    )

    expected = -len("sharp cat") * math.log(16)
    torch.testing.assert_close(log_probs, torch.tensor([expected], dtype=torch.float32))


def test_causal_lm_prompt_refiner_log_probs_can_reuse_rollout_input_text():
    refiner = object.__new__(CausalLMPromptRefiner)
    refiner.prompt_template = "ignored {prompt}"
    refiner.tokenizer = _CharTokenizer()
    refiner.transformer = _UniformLM(vocab_size=16)

    log_probs = refiner.compute_log_probs(
        original_prompts=["source"],
        refined_prompts=["xy"],
        metadata=[{"input_text": "cached prompt:"}],
    )

    expected = -len("xy") * math.log(16)
    torch.testing.assert_close(log_probs, torch.tensor([expected], dtype=torch.float32))


def test_resolve_torch_dtype_accepts_common_aliases():
    assert _resolve_torch_dtype(None) is None
    assert _resolve_torch_dtype("auto") is None
    assert _resolve_torch_dtype("bf16") is torch.bfloat16
    assert _resolve_torch_dtype("float16") is torch.float16
    assert _resolve_torch_dtype("fp32") is torch.float32
