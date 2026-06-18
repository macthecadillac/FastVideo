import pytest
import torch

from fastvideo.train.methods.rl.common import (
    PromptRefinementConfig,
    refine_prompt_batch,
)


def test_prompt_refinement_disabled_returns_original_prompts_without_mutating_batch():
    raw_batch = {
        "caption_text": ["a red cube"],
        "info_list": [{
            "prompt": "a red cube",
            "refined_prompt": "a glossy red cube",
        }],
    }

    refined_batch, result = refine_prompt_batch(raw_batch, PromptRefinementConfig())

    assert refined_batch is not raw_batch
    assert refined_batch["caption_text"] == ["a red cube"]
    assert refined_batch["info_list"][0]["prompt"] == "a red cube"
    assert result.original_prompts == ["a red cube"]
    assert result.prompts == ["a red cube"]
    assert result.refined_mask == [False]


def test_prompt_refinement_uses_dataset_column_and_preserves_prefix():
    raw_batch = {
        "caption_text": ["first prompt", "second prompt"],
        "info_list": [{
            "prompt": "first prompt",
            "refined_prompt": "first refined",
        }, {
            "prompt": "second prompt",
            "refined_prompt": "second refined",
        }],
    }
    config = PromptRefinementConfig.from_mapping({
        "enabled": True,
        "mode": "dataset_column",
        "num_original_prompts": 1,
    })

    refined_batch, result = refine_prompt_batch(raw_batch, config)

    assert refined_batch["caption_text"] == ["first prompt", "second refined"]
    assert refined_batch["info_list"][0]["prompt"] == "first prompt"
    assert refined_batch["info_list"][1]["prompt"] == "second refined"
    assert raw_batch["info_list"][1]["prompt"] == "second prompt"
    assert result.original_prompts == ["first prompt", "second prompt"]
    assert result.prompts == ["first prompt", "second refined"]
    assert result.refined_mask == [False, True]
    assert result.refined_count == 1


def test_prompt_refinement_template_mode_rewrites_caption_text():
    raw_batch = {
        "caption_text": ["a blue sphere"],
    }
    config = PromptRefinementConfig.from_mapping({
        "enabled": True,
        "mode": "template",
        "template": "cinematic render of {prompt}",
    })

    refined_batch, result = refine_prompt_batch(raw_batch, config)

    assert refined_batch["caption_text"] == ["cinematic render of a blue sphere"]
    assert result.refined_mask == [True]


def test_prompt_refinement_rejects_template_without_prompt_placeholder():
    with pytest.raises(ValueError, match="must contain"):
        PromptRefinementConfig.from_mapping({
            "enabled": True,
            "mode": "template",
            "template": "cinematic render",
        })


def test_prompt_refinement_model_mode_uses_refiner_and_preserves_original_prefix():

    class _PromptRefiner:

        def refine_prompts(self, *, prompts, raw_batch, config, generator):
            del raw_batch, config, generator
            return {
                "prompts": [f"refined {prompt}" for prompt in prompts],
                "log_probs": torch.tensor([-0.1, -0.2]),
                "metadata": [{"rank": idx} for idx, _ in enumerate(prompts)],
            }

    raw_batch = {
        "caption_text": ["first", "second", "third"],
    }
    config = PromptRefinementConfig.from_mapping({
        "enabled": True,
        "mode": "model",
        "num_original_prompts": 1,
    })

    refined_batch, result = refine_prompt_batch(raw_batch, config, prompt_refiner=_PromptRefiner())

    assert refined_batch["caption_text"] == ["first", "refined second", "refined third"]
    assert result.original_prompts == ["first", "second", "third"]
    assert result.prompts == ["first", "refined second", "refined third"]
    assert result.refined_mask == [False, True, True]
    assert result.policy_mask == [False, True, True]
    assert result.metadata == [{}, {"rank": 0}, {"rank": 1}]
    assert result.refiner_log_probs is not None
    torch.testing.assert_close(result.refiner_log_probs, torch.tensor([0.0, -0.1, -0.2]))


def test_prompt_refinement_model_mode_requires_refiner():
    config = PromptRefinementConfig.from_mapping({
        "enabled": True,
        "mode": "model",
    })

    with pytest.raises(ValueError, match="prompt_refiner"):
        refine_prompt_batch({"caption_text": ["prompt"]}, config)
