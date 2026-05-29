# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from fastvideo.pipelines.stages.denoising import DenoisingStage


def test_repeat_cfg_value_duplicates_batch_tensors_and_leaves_static_metadata() -> None:
    value = torch.arange(4).reshape(2, 2)
    image_embeds = [value]
    static_mask = [[[None]]]

    repeated_tensor = DenoisingStage._repeat_cfg_value(value)
    repeated_embeds = DenoisingStage._repeat_cfg_value(image_embeds)
    repeated_mask = DenoisingStage._repeat_cfg_value(static_mask)

    assert torch.equal(repeated_tensor, torch.cat([value, value], dim=0))
    assert torch.equal(repeated_embeds[0], torch.cat([value, value], dim=0))
    assert repeated_mask is static_mask


def test_merge_cfg_condition_kwargs_concats_uncond_then_cond() -> None:
    cond = {
        "encoder_hidden_states_2": [torch.full((1, 2), 2.0)],
        "encoder_attention_mask": [torch.ones(1, 2)],
    }
    uncond = {
        "encoder_hidden_states_2": [torch.full((1, 2), -1.0)],
        "encoder_attention_mask": [torch.zeros(1, 2)],
    }

    merged = DenoisingStage._merge_cfg_condition_kwargs(cond, uncond)

    assert torch.equal(merged["encoder_hidden_states_2"][0], torch.tensor([[-1.0, -1.0], [2.0, 2.0]]))
    assert torch.equal(merged["encoder_attention_mask"][0], torch.tensor([[0.0, 0.0], [1.0, 1.0]]))


def test_concat_cfg_value_rejects_mismatched_lists() -> None:
    with pytest.raises(Exception, match="list lengths differ"):
        DenoisingStage._concat_cfg_value([torch.zeros(1), torch.zeros(1)], [torch.zeros(1)], "prompt_embeds")


def test_run_batched_cfg_forward_uses_one_transformer_call() -> None:
    stage = object.__new__(DenoisingStage)
    calls = []

    latent_model_input = torch.zeros(1, 1, 1, 1, 1)
    prompt_embeds = [torch.full((1, 1), 3.0)]
    neg_prompt_embeds = [torch.full((1, 1), -2.0)]
    t_expand = torch.ones(1)

    def run_transformer_branch(
        model_input,
        embeds,
        timesteps_for_model,
        guidance_for_model,
        cond_kwargs,
        common_image_kwargs,
        common_action_kwargs,
        common_camera_kwargs,
        common_timestep_kwargs,
        is_cfg_negative,
    ):
        calls.append(
            {
                "model_input": model_input,
                "embeds": embeds,
                "timesteps": timesteps_for_model,
                "cond_kwargs": cond_kwargs,
                "is_cfg_negative": is_cfg_negative,
            })
        return embeds[0].reshape(2, 1, 1, 1, 1).to(model_input.dtype)

    noise_pred_uncond, noise_pred_text = stage._run_batched_cfg_forward(
        run_transformer_branch,
        latent_model_input,
        prompt_embeds,
        neg_prompt_embeds,
        t_expand,
        None,
        {"encoder_attention_mask": [torch.ones(1, 1)]},
        {"encoder_attention_mask": [torch.zeros(1, 1)]},
        {"encoder_hidden_states_image": [torch.ones(1, 1)]},
        {},
        {},
        {},
    )

    assert len(calls) == 1
    assert calls[0]["model_input"].shape[0] == 2
    assert torch.equal(calls[0]["embeds"][0], torch.tensor([[-2.0], [3.0]]))
    assert torch.equal(calls[0]["timesteps"], torch.tensor([1.0, 1.0]))
    assert torch.equal(calls[0]["cond_kwargs"]["encoder_attention_mask"][0], torch.tensor([[0.0], [1.0]]))
    assert calls[0]["is_cfg_negative"] is False
    assert torch.equal(noise_pred_uncond, torch.full((1, 1, 1, 1, 1), -2.0))
    assert torch.equal(noise_pred_text, torch.full((1, 1, 1, 1, 1), 3.0))
