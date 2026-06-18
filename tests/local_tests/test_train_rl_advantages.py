import pytest
import torch

from fastvideo.train.methods.rl.common import (
    GroupedAdvantageConfig,
    compute_clipped_grpo_policy_loss,
    compute_grouped_advantages,
    compute_multi_reward_advantages,
    repeat_advantages_over_timesteps,
)


def test_compute_grouped_advantages_normalizes_by_prompt():
    rewards = torch.tensor([1.0, 3.0, 2.0, 6.0])
    prompts = ["a", "a", "b", "b"]

    advantages = compute_grouped_advantages(
        rewards,
        prompts,
        config=GroupedAdvantageConfig(epsilon=1e-6),
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        atol=3e-6,
        rtol=0,
    )


def test_compute_grouped_advantages_preserves_timestep_shape():
    rewards = torch.tensor([
        [1.0, 4.0],
        [3.0, 2.0],
        [2.0, 8.0],
        [6.0, 4.0],
    ])
    prompts = ["a", "a", "b", "b"]

    advantages = compute_grouped_advantages(
        rewards,
        prompts,
        config=GroupedAdvantageConfig(epsilon=1e-6),
    )

    assert advantages.shape == (4, 2)
    torch.testing.assert_close(
        advantages,
        torch.tensor([
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
        ]),
        atol=2e-6,
        rtol=0,
    )


def test_compute_multi_reward_advantages_supports_weighted_reward_first_mode():
    rewards = {
        "pickscore": torch.tensor([1.0, 3.0, 2.0, 6.0]),
        "clipscore": torch.tensor([3.0, 1.0, 8.0, 4.0]),
    }
    prompts = ["a", "a", "b", "b"]

    advantages = compute_multi_reward_advantages(
        rewards,
        {
            "pickscore": 1.0,
            "clipscore": 0.5,
        },
        prompts,
        config=GroupedAdvantageConfig(epsilon=1e-6),
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        atol=3e-6,
        rtol=0,
    )


def test_compute_multi_reward_advantages_supports_weight_advantages_mode():
    rewards = {
        "pickscore": torch.tensor([1.0, 3.0, 2.0, 6.0]),
        "clipscore": torch.tensor([3.0, 1.0, 8.0, 4.0]),
    }
    prompts = ["a", "a", "b", "b"]

    advantages = compute_multi_reward_advantages(
        rewards,
        {
            "pickscore": 1.0,
            "clipscore": 0.5,
        },
        prompts,
        weight_advantages=True,
        config=GroupedAdvantageConfig(epsilon=1e-6),
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([-0.5, 0.5, -0.5, 0.5]),
        atol=3e-6,
        rtol=0,
    )


def test_repeat_advantages_over_timesteps_expands_one_dimensional_advantages():
    advantages = torch.tensor([1.0, -1.0])

    expanded = repeat_advantages_over_timesteps(advantages, 3)

    torch.testing.assert_close(
        expanded,
        torch.tensor([
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ]),
    )


def test_compute_clipped_grpo_policy_loss_uses_pessimistic_ratio_loss():
    log_probs = torch.log(torch.tensor([2.0, 2.0]))
    old_log_probs = torch.zeros(2)
    advantages = torch.tensor([1.0, -1.0])

    losses = compute_clipped_grpo_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        clip_range=0.1,
    )

    torch.testing.assert_close(losses["policy_loss"], torch.tensor(0.45), atol=1e-6, rtol=0)
    torch.testing.assert_close(losses["unclipped_policy_loss"], torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(losses["clip_frac"], torch.tensor(1.0))


def test_compute_clipped_grpo_policy_loss_validates_mask_selects_elements():
    with pytest.raises(ValueError, match="select at least one"):
        compute_clipped_grpo_policy_loss(
            torch.zeros(2),
            torch.zeros(2),
            torch.ones(2),
            clip_range=0.1,
            mask=torch.zeros(2, dtype=torch.bool),
        )
