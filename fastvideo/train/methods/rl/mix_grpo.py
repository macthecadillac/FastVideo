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
from fastvideo.train.utils.config import parse_betas
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler,
    clip_grad_norm_if_needed,
)


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
        self.prompt_refiner = role_models.get("prompt_refiner")
        self.old_prompt_refiner = role_models.get("old_prompt_refiner")
        self._prompt_clip_range = self._read_float("prompt_clip_range", self._clip_range)
        self._prompt_refiner_max_grad_norm = self._read_float("prompt_refiner_max_grad_norm", self._max_grad_norm)
        self._prompt_refiner_update_interval = self._read_int("prompt_refiner_update_interval", 1)
        self._prompt_refiner_optimizer: torch.optim.Optimizer | None = None
        self._prompt_refiner_lr_scheduler: Any | None = None
        if self._clip_range < 0.0:
            raise ValueError("method.clip_range must be non-negative")
        if self._prompt_clip_range < 0.0:
            raise ValueError("method.prompt_clip_range must be non-negative")
        if self._advantage_epsilon <= 0.0:
            raise ValueError("method.advantage_epsilon must be positive")
        if self._prompt_refiner_update_interval <= 0:
            raise ValueError("method.prompt_refiner_update_interval must be positive")
        if self._sampling_config.sde_noise_scale <= 0.0:
            raise ValueError("MixGRPO requires method.sampling.sde_noise_scale > 0")
        if self._num_train_timesteps() <= 0:
            raise ValueError("MixGRPO requires at least one SDE/log-prob training timestep")
        if self.old_prompt_refiner is not None and self.prompt_refiner is None:
            raise ValueError("models.old_prompt_refiner requires models.prompt_refiner")
        if self._prompt_refinement_config.enabled and self._prompt_refinement_config.mode == "model":
            if self.prompt_refiner is None:
                raise ValueError("PromptRL model refinement requires a models.prompt_refiner role")
            if not callable(getattr(self._prompt_rollout_refiner(), "refine_prompts", None)):
                raise ValueError("prompt_refiner rollout role must implement refine_prompts(...)")
        self._init_prompt_refiner_optimizer()

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        optimizers = dict(super()._optimizer_dict)
        if self._prompt_refiner_optimizer is not None:
            optimizers["prompt_refiner"] = self._prompt_refiner_optimizer
        return optimizers

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        schedulers = dict(super()._lr_scheduler_dict)
        if self._prompt_refiner_lr_scheduler is not None:
            schedulers["prompt_refiner"] = self._prompt_refiner_lr_scheduler
        return schedulers

    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        del iteration
        optimizers = [self._student_optimizer]
        if self._prompt_refiner_optimizer is not None:
            optimizers.append(self._prompt_refiner_optimizer)
        return optimizers

    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        del iteration
        schedulers = [self._student_lr_scheduler]
        if self._prompt_refiner_lr_scheduler is not None:
            schedulers.append(self._prompt_refiner_lr_scheduler)
        return schedulers

    def on_train_start(self) -> None:
        super().on_train_start()
        if self.prompt_refiner is not None and callable(getattr(self.prompt_refiner, "on_train_start", None)):
            self.prompt_refiner.on_train_start()
        if self.old_prompt_refiner is not None and callable(getattr(self.old_prompt_refiner, "on_train_start", None)):
            self.old_prompt_refiner.on_train_start()
        self._sync_old_prompt_refiner_from_prompt_refiner()

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
                raw_batch, prompt_refinement = refine_prompt_batch(
                    raw_batch,
                    self._prompt_refinement_config,
                    prompt_refiner=self._prompt_rollout_refiner(),
                    generator=self.cuda_generator,
                )
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
                    "prompt_policy_mask": prompt_refinement.policy_mask,
                    "prompt_policy_metadata": prompt_refinement.metadata,
                })
                if prompt_refinement.refiner_log_probs is not None:
                    sample_items[-1]["prompt_policy_log_probs"] = prompt_refinement.refiner_log_probs.detach()
                if prompt_refinement.old_refiner_log_probs is not None:
                    sample_items[-1]["prompt_policy_old_log_probs"] = (
                        prompt_refinement.old_refiner_log_probs.detach())
        return sample_items

    def _inner_train(
        self,
        sample_items: list[dict[str, Any]],
        advantages: torch.Tensor,
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        loss_map, metrics = super()._inner_train(sample_items, advantages, iteration)
        prompt_loss_map, prompt_metrics = self._maybe_train_prompt_refiner(sample_items, advantages, iteration)
        loss_map.update(prompt_loss_map)
        metrics.update(prompt_metrics)
        return loss_map, metrics

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
        policy_mask = [
            bool(selected)
            for item in sample_items
            for selected in item.get("prompt_policy_mask", [])
        ]
        if policy_mask:
            policy_count = sum(1 for selected in policy_mask if selected)
            metrics["prompt_refinement/policy_ratio"] = policy_count / len(policy_mask)
            metrics["prompt_refinement/policy_count"] = float(policy_count)
        return metrics

    def _compute_advantages(
        self,
        sample_items: list[dict[str, Any]],
        rewards: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        local_advantages = self._compute_sample_advantages(sample_items, rewards)
        return repeat_advantages_over_timesteps(local_advantages, self._num_train_timesteps())

    def _compute_sample_advantages(
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
        return global_advantages[start:end].to(first_reward.device)

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

    def _init_prompt_refiner_optimizer(self) -> None:
        if self.prompt_refiner is None or not bool(getattr(self.prompt_refiner, "_trainable", False)):
            return
        transformer = getattr(self.prompt_refiner, "transformer", None)
        if not isinstance(transformer, torch.nn.Module):
            raise ValueError("Trainable prompt_refiner roles must expose a torch.nn.Module transformer")
        params = [p for p in transformer.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Trainable prompt_refiner has no trainable transformer parameters")
        if not callable(getattr(self.prompt_refiner, "compute_log_probs", None)):
            raise ValueError("Trainable prompt_refiner must implement compute_log_probs(...)")

        learning_rate = self._read_float(
            "prompt_refiner_learning_rate",
            float(self.training_config.optimizer.learning_rate),
        )
        betas = self.training_config.optimizer.betas
        betas_raw = self.method_config.get("prompt_refiner_betas", None)
        if betas_raw is not None:
            betas = parse_betas(betas_raw, where="method.prompt_refiner_betas")
        self._prompt_refiner_optimizer, self._prompt_refiner_lr_scheduler = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=self.training_config.optimizer,
            loop_config=self.training_config.loop,
            learning_rate=learning_rate,
            betas=betas,
            scheduler_name=str(self.training_config.optimizer.lr_scheduler),
        )

    def _maybe_train_prompt_refiner(
        self,
        sample_items: list[dict[str, Any]],
        advantages: torch.Tensor,
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if self._prompt_refiner_optimizer is None or self.prompt_refiner is None:
            return {}, {}
        if not self._prompt_refinement_config.enabled or self._prompt_refinement_config.mode != "model":
            return {}, {}
        if iteration % self._prompt_refiner_update_interval != 0:
            return {}, {"prompt_refiner/update": 0.0}

        policy_batch = self._collect_prompt_policy_batch(sample_items, advantages)
        if policy_batch is None:
            return {}, {"prompt_refiner/update": 0.0, "prompt_refiner/num_samples": 0.0}

        transformer = getattr(self.prompt_refiner, "transformer", None)
        if isinstance(transformer, torch.nn.Module):
            transformer.train()
        self._prompt_refiner_optimizer.zero_grad(set_to_none=True)
        log_probs = self._compute_prompt_refiner_log_probs(
            self.prompt_refiner,
            policy_batch["source_prompts"],
            policy_batch["prompts"],
            policy_batch["metadata"],
        )
        old_log_probs = policy_batch["old_log_probs"]
        if old_log_probs is None:
            old_log_probs = self._compute_old_prompt_refiner_log_probs(policy_batch, log_probs)
        old_log_probs = old_log_probs.to(device=log_probs.device, dtype=log_probs.dtype)
        prompt_advantages = policy_batch["advantages"].to(device=log_probs.device, dtype=log_probs.dtype)
        policy_losses = compute_clipped_grpo_policy_loss(
            log_probs,
            old_log_probs,
            prompt_advantages,
            clip_range=self._prompt_clip_range,
        )
        policy_losses["policy_loss"].backward()
        grad_norm = self._clip_prompt_refiner_grads()
        self._prompt_refiner_optimizer.step()
        if self._prompt_refiner_lr_scheduler is not None:
            self._prompt_refiner_lr_scheduler.step()
        self._prompt_refiner_optimizer.zero_grad(set_to_none=True)

        loss_map = {
            "prompt_refiner_policy_loss": policy_losses["policy_loss"].detach(),
            "prompt_refiner_approx_kl": policy_losses["approx_kl"].detach(),
        }
        metrics = {
            "prompt_refiner/update": 1.0,
            "prompt_refiner/num_samples": float(len(policy_batch["prompts"])),
            "prompt_refiner/policy_loss": self._mean_scalar_across_ranks(policy_losses["policy_loss"]),
            "prompt_refiner/approx_kl": self._mean_scalar_across_ranks(policy_losses["approx_kl"]),
            "prompt_refiner/clip_frac": self._mean_scalar_across_ranks(policy_losses["clip_frac"]),
            "prompt_refiner/grad_norm": float(grad_norm),
        }
        return loss_map, metrics

    def _collect_prompt_policy_batch(
        self,
        sample_items: list[dict[str, Any]],
        advantages: torch.Tensor,
    ) -> dict[str, Any] | None:
        sample_advantages = advantages[:, 0] if advantages.ndim == 2 else advantages
        source_prompts: list[str] = []
        prompts: list[str] = []
        metadata: list[Any] = []
        selected_advantages: list[torch.Tensor] = []
        old_log_prob_chunks: list[torch.Tensor] = []
        has_old_log_probs = True
        cursor = 0

        for item in sample_items:
            item_prompts = list(item["prompts"])
            item_sources = list(item.get("source_prompts", item_prompts))
            item_metadata = list(item.get("prompt_policy_metadata", [{} for _ in item_prompts]))
            item_mask = [bool(value) for value in item.get("prompt_policy_mask", [False for _ in item_prompts])]
            old_log_probs = item.get("prompt_policy_old_log_probs", item.get("prompt_policy_log_probs"))
            old_log_probs = self._prompt_log_probs_to_vector(old_log_probs) if old_log_probs is not None else None
            has_old_log_probs = has_old_log_probs and old_log_probs is not None
            for idx, selected in enumerate(item_mask):
                if selected:
                    source_prompts.append(item_sources[idx])
                    prompts.append(item_prompts[idx])
                    metadata.append(item_metadata[idx])
                    selected_advantages.append(sample_advantages[cursor + idx].detach())
                    if old_log_probs is not None:
                        old_log_prob_chunks.append(old_log_probs[idx].detach())
            cursor += len(item_prompts)

        if not prompts:
            return None
        device = self.student.device
        return {
            "source_prompts": source_prompts,
            "prompts": prompts,
            "metadata": metadata,
            "advantages": torch.stack(selected_advantages).to(device=device, dtype=torch.float32),
            "old_log_probs": (torch.stack(old_log_prob_chunks).to(device=device, dtype=torch.float32)
                              if has_old_log_probs else None),
        }

    def _compute_old_prompt_refiner_log_probs(
        self,
        policy_batch: dict[str, Any],
        current_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        if self.old_prompt_refiner is None:
            return current_log_probs.detach()
        with torch.no_grad():
            return self._compute_prompt_refiner_log_probs(
                self.old_prompt_refiner,
                policy_batch["source_prompts"],
                policy_batch["prompts"],
                policy_batch["metadata"],
            ).detach()

    def _compute_prompt_refiner_log_probs(
        self,
        prompt_refiner: Any,
        source_prompts: list[str],
        prompts: list[str],
        metadata: list[Any],
    ) -> torch.Tensor:
        compute_fn = getattr(prompt_refiner, "compute_log_probs", None)
        if not callable(compute_fn):
            raise RuntimeError("prompt_refiner must implement compute_log_probs(...) to train PromptRL policy")
        output = compute_fn(
            original_prompts=source_prompts,
            refined_prompts=prompts,
            metadata=metadata,
        )
        if isinstance(output, dict):
            output = output.get("log_probs")
        return self._prompt_log_probs_to_vector(output)

    @staticmethod
    def _prompt_log_probs_to_vector(log_probs: Any) -> torch.Tensor:
        if log_probs is None:
            raise ValueError("prompt log probabilities are missing")
        tensor = log_probs if torch.is_tensor(log_probs) else torch.as_tensor(log_probs, dtype=torch.float32)
        if tensor.ndim == 0:
            raise ValueError("prompt log probabilities must include a batch dimension")
        if tensor.ndim == 1:
            return tensor
        return tensor.sum(dim=tuple(range(1, tensor.ndim)))

    def _clip_prompt_refiner_grads(self) -> float:
        if self._prompt_refiner_max_grad_norm <= 0.0 or self.prompt_refiner is None:
            return 0.0
        transformer = getattr(self.prompt_refiner, "transformer", None)
        if not isinstance(transformer, torch.nn.Module):
            return 0.0
        return clip_grad_norm_if_needed(transformer, self._prompt_refiner_max_grad_norm)

    def _prompt_rollout_refiner(self) -> Any | None:
        return self.old_prompt_refiner if self.old_prompt_refiner is not None else self.prompt_refiner

    def _sync_old_prompt_refiner_from_prompt_refiner(self) -> None:
        if self.prompt_refiner is None or self.old_prompt_refiner is None:
            return
        source = getattr(self.prompt_refiner, "transformer", None)
        target = getattr(self.old_prompt_refiner, "transformer", None)
        if not isinstance(source, torch.nn.Module) or not isinstance(target, torch.nn.Module):
            return
        with torch.no_grad():
            for src, tgt in zip(source.parameters(), target.parameters(), strict=True):
                tgt.data.copy_(src.detach().data)

    def _update_old_model(self, iteration: int) -> None:
        super()._update_old_model(iteration)
        if self.prompt_refiner is None or self.old_prompt_refiner is None:
            return
        source = getattr(self.prompt_refiner, "transformer", None)
        target = getattr(self.old_prompt_refiner, "transformer", None)
        if not isinstance(source, torch.nn.Module) or not isinstance(target, torch.nn.Module):
            return
        decay = self._return_decay(iteration, self._decay_type)
        with torch.no_grad():
            for src, tgt in zip(source.parameters(), target.parameters(), strict=True):
                tgt.data.copy_(tgt.detach().data * decay + src.detach().data * (1.0 - decay))
