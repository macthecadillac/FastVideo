from types import SimpleNamespace

import torch

from fastvideo.train.methods.rl.common import DiffusionSampler, SamplingConfig
from fastvideo.train.methods.rl.mix_grpo import MixGRPOMethod


class _FakeScheduler:

    def __init__(self):
        self.num_train_timesteps = 1000
        self.timesteps = torch.empty(0)
        self.sigmas = torch.empty(0)

    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None, sigmas=None):
        if timesteps is not None:
            self.timesteps = torch.tensor(timesteps, device=device, dtype=torch.float32)
        else:
            self.timesteps = torch.linspace(1000, 0, int(num_inference_steps), device=device)
        if sigmas is not None:
            self.sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
        else:
            self.sigmas = torch.cat([self.timesteps / 1000.0, torch.zeros(1, device=device)])


class _FakeRole:

    def __init__(self, value: float = 0.0):
        self.noise_scheduler = _FakeScheduler()
        self.device = torch.device("cpu")
        self.transformer = SimpleNamespace()
        self.value = float(value)

    def predict_noise(self, noisy_latents, timestep, batch, *, conditional, attn_kind):
        del timestep, conditional, attn_kind
        assert batch.conditional_dict is not None
        return torch.full_like(noisy_latents, self.value)


class _FakePromptRefiner:

    def __init__(self):
        self._trainable = True
        self.transformer = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.transformer.weight)

    def compute_log_probs(self, *, original_prompts, refined_prompts, metadata):
        del original_prompts, metadata
        features = torch.tensor([[float(len(prompt))] for prompt in refined_prompts])
        return self.transformer(features).squeeze(1)


def test_mixgrpo_num_train_timesteps_uses_sde_mask():
    method = object.__new__(MixGRPOMethod)
    method._sample_steps = 6
    method._sampling_config = SamplingConfig(
        num_steps=6,
        trajectory="mixed_ode_sde",
        sde_window_start=2,
        sde_window_size=3,
    )

    assert method._num_train_timesteps() == 3


def test_mixgrpo_compute_advantages_weights_rewards_before_normalizing():
    method = object.__new__(MixGRPOMethod)
    method._sample_steps = 4
    method._sampling_config = SamplingConfig(
        num_steps=4,
        trajectory="mixed_ode_sde",
        sde_window_start=1,
        sde_window_size=2,
    )
    method._reward_fn_config = {
        "pickscore": 1.0,
        "clipscore": 0.5,
    }
    method._weight_advantages = False
    method._advantage_epsilon = 1e-6
    sample_items = [{
        "prompts": ["a", "a"],
    }, {
        "prompts": ["b", "b"],
    }]
    rewards = {
        "pickscore": torch.tensor([1.0, 3.0, 2.0, 6.0]),
        "clipscore": torch.tensor([3.0, 1.0, 8.0, 4.0]),
    }

    advantages = method._compute_advantages(sample_items, rewards)

    torch.testing.assert_close(
        advantages,
        torch.tensor([
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
        ]),
        atol=3e-6,
        rtol=0,
    )


def test_mixgrpo_compute_advantages_can_weight_normalized_reward_heads():
    method = object.__new__(MixGRPOMethod)
    method._sample_steps = 4
    method._sampling_config = SamplingConfig(
        num_steps=4,
        trajectory="mixed_ode_sde",
        sde_window_start=1,
        sde_window_size=2,
    )
    method._reward_fn_config = {
        "pickscore": 1.0,
        "clipscore": 0.5,
    }
    method._weight_advantages = True
    method._advantage_epsilon = 1e-6
    sample_items = [{
        "prompts": ["a", "a"],
    }, {
        "prompts": ["b", "b"],
    }]
    rewards = {
        "pickscore": torch.tensor([1.0, 3.0, 2.0, 6.0]),
        "clipscore": torch.tensor([3.0, 1.0, 8.0, 4.0]),
    }

    advantages = method._compute_advantages(sample_items, rewards)

    torch.testing.assert_close(
        advantages,
        torch.tensor([
            [-0.5, -0.5],
            [0.5, 0.5],
            [-0.5, -0.5],
            [0.5, 0.5],
        ]),
        atol=3e-6,
        rtol=0,
    )


def test_mixgrpo_training_timestep_loss_returns_finite_grpo_terms():
    method = object.__new__(MixGRPOMethod)
    method.student = _FakeRole(value=0.0)
    method.reference = _FakeRole(value=0.0)
    method._sampling_config = SamplingConfig(
        num_steps=3,
        trajectory="mixed_ode_sde",
        sde_window_start=1,
        sde_window_size=1,
        sde_noise_scale=0.7,
    )
    method._sampler = DiffusionSampler(method._sampling_config)
    method._clip_range = 0.1
    method._kl_beta = 0.0
    sample = {
        "encoder_hidden_states": torch.zeros(2, 1, 4),
        "encoder_attention_mask": torch.ones(2, 1),
        "latents": torch.zeros(2, 1, 1, 1, 2, 2),
        "next_latents": torch.zeros(2, 1, 1, 1, 2, 2),
        "timesteps": torch.tensor([[500.0], [500.0]]),
        "log_probs": torch.zeros(2, 1),
    }

    losses, backward_ctx = method._training_timestep_loss(sample, torch.tensor([1.0, -1.0]), 0)

    assert set(losses) == {
        "total_loss",
        "policy_loss",
        "unclipped_policy_loss",
        "approx_kl",
        "clip_frac",
        "kl_loss",
        "log_prob",
        "old_log_prob",
    }
    assert torch.isfinite(losses["total_loss"])
    assert torch.equal(backward_ctx[0], sample["timesteps"][:, 0])


def test_mixgrpo_reward_diagnostics_include_prompt_refinement_stats():
    method = object.__new__(MixGRPOMethod)
    method._trained_prompt_hashes = set()
    sample_items = [{
        "prompts": ["a", "a"],
        "prompt_refined_mask": [False, True],
        "prompt_policy_mask": [False, False],
    }, {
        "prompts": ["b", "b"],
        "prompt_refined_mask": [True, True],
        "prompt_policy_mask": [False, False],
    }]
    rewards = {"avg": torch.tensor([1.0, 3.0, 2.0, 6.0])}

    metrics = method._reward_diagnostic_metrics(sample_items, rewards)

    assert metrics["prompt_refinement/refined_ratio"] == 0.75
    assert metrics["prompt_refinement/refined_count"] == 3.0
    assert metrics["prompt_refinement/policy_ratio"] == 0.0
    assert metrics["prompt_refinement/policy_count"] == 0.0


def test_mixgrpo_prompt_refiner_policy_update_steps_optimizer():
    method = object.__new__(MixGRPOMethod)
    prompt_refiner = _FakePromptRefiner()
    method.prompt_refiner = prompt_refiner
    method.old_prompt_refiner = None
    method.student = SimpleNamespace(device=torch.device("cpu"))
    method._prompt_refinement_config = SimpleNamespace(enabled=True, mode="model")
    method._prompt_refiner_update_interval = 1
    method._prompt_clip_range = 0.2
    method._prompt_refiner_max_grad_norm = 0.0
    method._prompt_refiner_optimizer = torch.optim.SGD(prompt_refiner.transformer.parameters(), lr=0.1)
    method._prompt_refiner_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        method._prompt_refiner_optimizer,
        lr_lambda=lambda _: 1.0,
    )
    sample_items = [{
        "source_prompts": ["a", "bb"],
        "prompts": ["aa", "bbbb"],
        "prompt_policy_mask": [True, True],
        "prompt_policy_old_log_probs": torch.zeros(2),
        "prompt_policy_metadata": [{}, {}],
    }]
    advantages = torch.tensor([[1.0], [1.0]])

    old_weight = prompt_refiner.transformer.weight.detach().clone()
    loss_map, metrics = method._maybe_train_prompt_refiner(sample_items, advantages, iteration=1)

    assert set(loss_map) == {
        "prompt_refiner_policy_loss",
        "prompt_refiner_approx_kl",
    }
    assert metrics["prompt_refiner/update"] == 1.0
    assert metrics["prompt_refiner/num_samples"] == 2.0
    assert prompt_refiner.transformer.weight.item() > old_weight.item()
