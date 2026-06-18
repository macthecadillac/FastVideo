# SPDX-License-Identifier: Apache-2.0
"""Reusable RL training primitives."""

from fastvideo.train.methods.rl.common.advantages import (
    GroupedAdvantageConfig,
    compute_clipped_grpo_policy_loss,
    compute_grouped_advantages,
    compute_multi_reward_advantages,
    repeat_advantages_over_timesteps,
)
from fastvideo.train.methods.rl.common.sampling import (
    DiffusionSampler,
    SamplingConfig,
    SamplingResult,
    sde_step_mask,
)
from fastvideo.train.methods.rl.common.prompt_sampling import (
    KRepeatSample,
    distributed_k_repeat_indices,
)
from fastvideo.train.methods.rl.common.validation import (
    RLValidationConfig,
    media_to_video_array,
    validation_caption,
    validation_shard_indices,
)

__all__ = [
    "DiffusionSampler",
    "GroupedAdvantageConfig",
    "KRepeatSample",
    "RLValidationConfig",
    "SamplingConfig",
    "SamplingResult",
    "compute_clipped_grpo_policy_loss",
    "compute_grouped_advantages",
    "compute_multi_reward_advantages",
    "distributed_k_repeat_indices",
    "media_to_video_array",
    "repeat_advantages_over_timesteps",
    "sde_step_mask",
    "validation_caption",
    "validation_shard_indices",
]
