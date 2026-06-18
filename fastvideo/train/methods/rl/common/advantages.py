# SPDX-License-Identifier: Apache-2.0
"""Advantage and policy-loss helpers for GRPO-style RL methods."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GroupedAdvantageConfig:
    """Configuration for per-prompt GRPO advantage normalization."""

    epsilon: float = 1e-4
    clip_min: float | None = None
    clip_max: float | None = None


def compute_grouped_advantages(
    rewards: torch.Tensor,
    prompts: Sequence[str],
    *,
    config: GroupedAdvantageConfig | None = None,
) -> torch.Tensor:
    """Normalize rewards within each prompt group.

    ``rewards`` may be ``[N]`` or ``[N, ...]``. Normalization is applied over
    the first dimension for every prompt group, preserving any trailing
    timestep/reward-head dimensions.
    """
    if not torch.is_tensor(rewards):
        raise TypeError(f"rewards must be a torch.Tensor, got {type(rewards).__name__}")
    if rewards.ndim < 1:
        raise ValueError("rewards must have at least one dimension")
    if int(rewards.shape[0]) != len(prompts):
        raise ValueError(f"reward batch size ({rewards.shape[0]}) must match prompt count ({len(prompts)})")

    cfg = config or GroupedAdvantageConfig()
    if cfg.epsilon <= 0.0:
        raise ValueError("GroupedAdvantageConfig.epsilon must be positive")
    advantages = torch.empty_like(rewards, dtype=torch.float32)
    rewards_f = rewards.detach().float()

    prompt_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, prompt in enumerate(prompts):
        prompt_to_indices[str(prompt)].append(idx)

    for indices in prompt_to_indices.values():
        index_tensor = torch.tensor(indices, device=rewards.device, dtype=torch.long)
        group_rewards = rewards_f.index_select(0, index_tensor)
        group_mean = group_rewards.mean(dim=0, keepdim=True)
        group_std = group_rewards.std(dim=0, unbiased=False, keepdim=True)
        group_advantages = (group_rewards - group_mean) / (group_std + float(cfg.epsilon))
        advantages.index_copy_(0, index_tensor, group_advantages)

    if cfg.clip_min is not None or cfg.clip_max is not None:
        clip_min = -float("inf") if cfg.clip_min is None else float(cfg.clip_min)
        clip_max = float("inf") if cfg.clip_max is None else float(cfg.clip_max)
        advantages = torch.clamp(advantages, clip_min, clip_max)
    return advantages


def compute_multi_reward_advantages(
    rewards: Mapping[str, torch.Tensor],
    reward_weights: Mapping[str, float],
    prompts: Sequence[str],
    *,
    weight_advantages: bool = False,
    config: GroupedAdvantageConfig | None = None,
) -> torch.Tensor:
    """Compute GRPO advantages from one or more reward heads.

    When ``weight_advantages`` is false, reward heads are weighted first and
    the weighted sum is normalized. When true, each reward head is normalized
    independently and weighted advantages are summed. This matches the two
    composition modes used by GenRL-style GRPO training.
    """
    if not reward_weights:
        raise ValueError("reward_weights must contain at least one reward")

    missing = sorted(set(reward_weights) - set(rewards))
    if missing:
        raise ValueError(f"Missing reward tensor(s): {missing}")

    if weight_advantages:
        total: torch.Tensor | None = None
        for name, weight in reward_weights.items():
            reward_advantages = compute_grouped_advantages(
                rewards[name],
                prompts,
                config=config,
            )
            weighted = reward_advantages * float(weight)
            total = weighted if total is None else total.to(weighted.device) + weighted
        assert total is not None
        return total

    weighted_rewards: torch.Tensor | None = None
    for name, weight in reward_weights.items():
        reward = rewards[name].detach().float() * float(weight)
        weighted_rewards = reward if weighted_rewards is None else weighted_rewards.to(reward.device) + reward
    assert weighted_rewards is not None
    return compute_grouped_advantages(
        weighted_rewards,
        prompts,
        config=config,
    )


def repeat_advantages_over_timesteps(
    advantages: torch.Tensor,
    num_timesteps: int,
) -> torch.Tensor:
    """Expand per-sample advantages to ``[N, num_timesteps]``."""
    num_timesteps = int(num_timesteps)
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if advantages.ndim != 1:
        raise ValueError(f"advantages must have shape [N], got {tuple(advantages.shape)}")
    return advantages.unsqueeze(1).repeat(1, num_timesteps)


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return values.mean()
    if mask.shape != values.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match value shape {tuple(values.shape)}")
    weights = mask.to(device=values.device, dtype=values.dtype)
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 0.0:
        raise ValueError("mask must select at least one element")
    return (values * weights).sum() / denom


def compute_clipped_grpo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the PPO-style clipped GRPO policy loss."""
    if log_probs.shape != old_log_probs.shape:
        raise ValueError("log_probs and old_log_probs must have the same shape "
                         f"({tuple(log_probs.shape)} vs {tuple(old_log_probs.shape)})")
    if advantages.shape != log_probs.shape:
        raise ValueError("advantages must have the same shape as log_probs "
                         f"({tuple(advantages.shape)} vs {tuple(log_probs.shape)})")
    clip_range = float(clip_range)
    if clip_range < 0.0:
        raise ValueError("clip_range must be non-negative")

    log_ratio = log_probs - old_log_probs.detach()
    ratio = torch.exp(log_ratio)
    advantages = advantages.detach().to(device=log_probs.device, dtype=log_probs.dtype)
    unclipped_loss = -advantages * ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    clipped_loss = -advantages * clipped_ratio
    per_token_loss = torch.maximum(unclipped_loss, clipped_loss)
    clipped = (torch.abs(ratio - 1.0) > clip_range)

    return {
        "policy_loss": _masked_mean(per_token_loss, mask),
        "unclipped_policy_loss": _masked_mean(unclipped_loss, mask),
        "approx_kl": _masked_mean(0.5 * log_ratio.square(), mask),
        "clip_frac": _masked_mean(clipped.to(dtype=log_probs.dtype), mask),
    }
