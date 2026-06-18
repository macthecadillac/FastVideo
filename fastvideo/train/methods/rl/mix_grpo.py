# SPDX-License-Identifier: Apache-2.0
"""MixGRPO policy optimization method for diffusion models."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.rl.common import (
    GroupedAdvantageConfig,
    PromptRefinementConfig,
    compute_clipped_grpo_policy_loss,
    compute_multi_reward_advantages,
    flow_sde_step_with_logprob,
    refine_prompt_batch,
    repeat_advantages_over_timesteps,
    sde_step_mask,
)
from fastvideo.train.methods.rl.diffusion_nft import DiffusionNFTMethod


class MixGRPOMethod(DiffusionNFTMethod):
    """GenRL-style MixGRPO built on the DiffusionNFT RL scaffold.

    PR #1450 already provides FastVideo's method-managed RL loop, prompt
    sampling, reward scoring, validation, and optimizer handling. MixGRPO keeps
    that scaffold but trains with a clipped GRPO objective on stochastic
    SDE-window transitions collected during sampling.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        self._clip_range = self._read_float("clip_range", 1.0e-4)
        self._advantage_epsilon = self._read_float("advantage_epsilon", 1.0e-4)
        self._weight_advantages = self._coerce_bool(
            self.method_config.get("weight_advantages", False),
            where="method.weight_advantages",
        )
        self._prompt_refinement_config = PromptRefinementConfig.from_mapping(
            self.method_config.get("prompt_refinement"))
        if self._clip_range < 0.0:
            raise ValueError("method.clip_range must be non-negative")
        if self._advantage_epsilon <= 0.0:
            raise ValueError("method.advantage_epsilon must be positive")
        if self._sampling_config.sde_noise_scale <= 0.0:
            raise ValueError("MixGRPO requires method.sampling.sde_noise_scale > 0")
        if self._num_train_timesteps() <= 0:
            raise ValueError("MixGRPO requires at least one SDE/log-prob training timestep")

    def _sample_epoch(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> list[dict[str, Any]]:
        self.student.transformer.eval()
        self.old.transformer.eval()
        sample_items: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch_idx in tqdm(
                    range(self._num_batches_per_epoch),
                    desc=f"MixGRPO step {iteration}: sampling",
                    position=1,
                    leave=False,
                    disable=not self._show_terminal_progress(),
            ):
                raw_batch = self._sample_prompt_batch(data_stream, iteration, batch_idx)
                raw_batch, prompt_refinement = refine_prompt_batch(raw_batch, self._prompt_refinement_config)
                prompts = prompt_refinement.prompts
                batch = self.student.prepare_batch(
                    raw_batch,
                    generator=self.cuda_generator,
                    latents_source="zeros",
                )
                sampling_result = self._sampler.sample_with_log_probs(
                    self.old,
                    batch,
                    generator=self.cuda_generator,
                )
                trace = sampling_result.trace
                if trace is None:
                    raise RuntimeError("MixGRPO sampling produced no SDE/log-prob transitions")
                latents_clean = sampling_result.latents
                media = self.student.decode_latents(latents_clean)
                sample_items.append({
                    "encoder_hidden_states": batch.encoder_hidden_states.detach(),
                    "encoder_attention_mask": batch.encoder_attention_mask.detach(),
                    "latents": trace.latents.detach(),
                    "next_latents": trace.next_latents.detach(),
                    "timesteps": trace.timesteps.detach(),
                    "log_probs": trace.log_probs.detach(),
                    "step_indices": trace.step_indices.detach(),
                    "latents_clean": latents_clean.detach(),
                    "media": media.detach().cpu(),
                    "prompts": prompts,
                    "source_prompts": prompt_refinement.original_prompts,
                    "prompt_refined_mask": prompt_refinement.refined_mask,
                })
        return sample_items

    def _reward_diagnostic_metrics(
        self,
        sample_items: list[dict[str, Any]],
        rewards: dict[str, torch.Tensor],
    ) -> dict[str, LogScalar]:
        metrics = super()._reward_diagnostic_metrics(sample_items, rewards)
        refined_mask = [
            bool(refined)
            for item in sample_items
            for refined in item.get("prompt_refined_mask", [])
        ]
        if refined_mask:
            refined_count = sum(1 for refined in refined_mask if refined)
            metrics["prompt_refinement/refined_ratio"] = refined_count / len(refined_mask)
            metrics["prompt_refinement/refined_count"] = float(refined_count)
        return metrics

    def _compute_advantages(
        self,
        sample_items: list[dict[str, Any]],
        rewards: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        local_prompts = [prompt for item in sample_items for prompt in item["prompts"]]
        local_count = len(local_prompts)
        if local_count <= 0:
            raise RuntimeError("MixGRPO requires at least one sampled prompt")

        gathered_prompts = self._gather_prompts(local_prompts)
        gathered_rewards = {
            name: self._gather_tensor(rewards[name].detach().float())
            for name in self._reward_fn_config
        }
        first_reward = next(iter(gathered_rewards.values()))
        if len(gathered_prompts) != int(first_reward.shape[0]):
            raise RuntimeError("Gathered prompt count does not match gathered rewards")

        global_advantages = compute_multi_reward_advantages(
            gathered_rewards,
            self._reward_fn_config,
            gathered_prompts,
            weight_advantages=self._weight_advantages,
            config=GroupedAdvantageConfig(epsilon=self._advantage_epsilon),
        )

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        start = rank * local_count
        end = start + local_count
        local_advantages = global_advantages[start:end].to(first_reward.device)
        return repeat_advantages_over_timesteps(local_advantages, self._num_train_timesteps())

    def _training_timestep_loss(
        self,
        sample: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        timestep_idx: int,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, Any]]:
        x_t = sample["latents"][:, timestep_idx]
        next_latents = sample["next_latents"][:, timestep_idx]
        timestep = sample["timesteps"][:, timestep_idx].to(device=x_t.device)
        old_log_probs = sample["log_probs"][:, timestep_idx]
        batch = self._make_mixgrpo_training_batch(sample, timestep, x_t)
        scheduler = self._sampler._prepare_scheduler(self.student, x_t.device)

        forward_prediction = self.student.predict_noise(
            x_t,
            timestep,
            batch,
            conditional=True,
            attn_kind="dense",
        )
        _, log_probs, prev_sample_mean, std_dev_t, dt_sqrt, _, _ = flow_sde_step_with_logprob(
            scheduler,
            forward_prediction,
            timestep,
            x_t,
            noise_level=self._sampling_config.sde_noise_scale,
            prev_sample=next_latents,
        )

        policy_losses = compute_clipped_grpo_policy_loss(
            log_probs,
            old_log_probs,
            advantages,
            clip_range=self._clip_range,
        )

        kl_loss = torch.zeros((), device=x_t.device, dtype=torch.float32)
        if self._kl_beta > 0.0:
            with torch.no_grad():
                ref_prediction = self.reference.predict_noise(
                    x_t,
                    timestep,
                    batch,
                    conditional=True,
                    attn_kind="dense",
                )
                _, _, ref_prev_sample_mean, _, ref_dt_sqrt, _, _ = flow_sde_step_with_logprob(
                    scheduler,
                    ref_prediction,
                    timestep,
                    x_t,
                    noise_level=self._sampling_config.sde_noise_scale,
                    prev_sample=next_latents,
                )
            kl_denom = (std_dev_t * ref_dt_sqrt).square()
            kl_loss = ((prev_sample_mean - ref_prev_sample_mean).square() /
                       (2.0 * kl_denom)).mean(dim=tuple(range(1, prev_sample_mean.ndim))).mean()

        total_loss = policy_losses["policy_loss"] + self._kl_beta * kl_loss
        losses = {
            "total_loss": total_loss,
            "policy_loss": policy_losses["policy_loss"],
            "unclipped_policy_loss": policy_losses["unclipped_policy_loss"],
            "approx_kl": policy_losses["approx_kl"],
            "clip_frac": policy_losses["clip_frac"],
            "kl_loss": kl_loss,
            "log_prob": log_probs.detach().mean(),
            "old_log_prob": old_log_probs.detach().mean(),
        }
        return losses, (batch.timesteps, batch.attn_metadata)

    def _collate_samples(self, sample_items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        keys = [
            "encoder_hidden_states",
            "encoder_attention_mask",
            "latents",
            "next_latents",
            "timesteps",
            "log_probs",
        ]
        return {key: torch.cat([item[key] for item in sample_items], dim=0).to(self.student.device) for key in keys}

    @staticmethod
    def _make_mixgrpo_training_batch(
        sample: dict[str, torch.Tensor],
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> TrainingBatch:
        batch = TrainingBatch()
        batch.encoder_hidden_states = sample["encoder_hidden_states"]
        batch.encoder_attention_mask = sample["encoder_attention_mask"]
        batch.conditional_dict = {
            "encoder_hidden_states": batch.encoder_hidden_states,
            "encoder_attention_mask": batch.encoder_attention_mask,
        }
        batch.timesteps = timestep
        batch.raw_latent_shape = tuple(latents.shape)
        return batch

    def _num_train_timesteps(self) -> int:
        schedule_len = self._sample_steps
        if self._sampling_config.timesteps is not None:
            schedule_len = len(self._sampling_config.timesteps)
        elif self._sampling_config.sigmas is not None:
            schedule_len = len(self._sampling_config.sigmas)
        mask = sde_step_mask(self._sampling_config, schedule_len)
        return int(mask.sum().item())
