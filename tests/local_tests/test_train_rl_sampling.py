import pytest
import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.rl.common import (
    DiffusionSampler,
    SamplingConfig,
    distributed_k_repeat_indices,
    media_to_video_array,
    sde_step_mask,
    validation_caption,
    validation_shard_indices,
)
from fastvideo.train.utils.config import load_run_config


class _FakeScheduler:

    def __init__(self):
        self.num_train_timesteps = 1000
        self.set_timesteps_calls = []
        self.timesteps = torch.empty(0)
        self.sigmas = torch.empty(0)
        self.step_calls = 0

    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None, sigmas=None):
        self.set_timesteps_calls.append({
            "num_inference_steps": num_inference_steps,
            "timesteps": timesteps,
            "sigmas": sigmas,
        })
        if timesteps is not None:
            self.timesteps = torch.tensor(timesteps, device=device, dtype=torch.float32)
        else:
            self.timesteps = torch.linspace(1000, 0, int(num_inference_steps), device=device)
        if sigmas is not None:
            self.sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
        else:
            self.sigmas = torch.cat([self.timesteps / 1000.0, torch.zeros(1, device=device)])

    def step(self, model_output, timestep, sample, return_dict=False):
        del timestep
        self.step_calls += 1
        prev = sample + model_output
        return (prev, ) if not return_dict else {"prev_sample": prev}


class _FakeModel:

    def __init__(self):
        self.noise_scheduler = _FakeScheduler()
        self.add_noise_calls = 0
        self.add_noise_inputs = []
        self.predict_noise_calls = 0
        self.predict_x0_calls = 0
        self.timestep_shapes = []

    def predict_noise(self, noisy_latents, timestep, batch, *, conditional, attn_kind):
        del conditional, attn_kind
        self.predict_noise_calls += 1
        self.timestep_shapes.append(tuple(timestep.shape))
        assert batch.timesteps is timestep
        return torch.zeros_like(noisy_latents)

    def predict_x0(self, noisy_latents, timestep, batch, *, conditional, attn_kind):
        del conditional, attn_kind
        self.predict_x0_calls += 1
        self.timestep_shapes.append(tuple(timestep.shape))
        assert batch.timesteps is timestep
        return noisy_latents

    def add_noise(self, clean_latents, noise, timestep):
        del timestep
        self.add_noise_calls += 1
        self.add_noise_inputs.append(noise)
        return clean_latents + noise


def _batch():
    batch = TrainingBatch()
    batch.latents = torch.zeros(2, 1, 3, 4, 4)
    return batch


def test_sampler_preserves_latent_dtype():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=4))
    batch = _batch()
    batch.latents = batch.latents.to(torch.bfloat16)

    result = sampler.sample(model, batch, generator=torch.Generator().manual_seed(0))

    assert result.latents.dtype is torch.bfloat16


def test_sampler_uses_scheduler_generated_timesteps_by_default():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=4))

    result = sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert result.timesteps.tolist() == [1000.0, 666.6666259765625, 333.3333435058594, 0.0]


def test_sampler_honors_explicit_timestep_override():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=3, timesteps=[900, 300, 10]))

    result = sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert result.timesteps.tolist() == [900.0, 300.0, 10.0]


def test_sampler_honors_explicit_timesteps_without_matching_num_steps():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(timesteps=[900, 300, 10]))

    result = sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert result.timesteps.tolist() == [900.0, 300.0, 10.0]
    assert model.noise_scheduler.set_timesteps_calls == []


def test_sampling_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unsupported method.sampling key"):
        SamplingConfig.from_mapping({"solver": "dpm2"})


def test_sampling_config_accepts_mixgrpo_window():
    cfg = SamplingConfig.from_mapping({
        "trajectory": "mixed_ode_sde",
        "num_steps": 5,
        "sde_window_start": 1,
        "sde_window_size": 2,
        "sde_noise_scale": 0.0,
    })

    assert cfg.trajectory == "mixed_ode_sde"
    assert cfg.sde_window_start == 1
    assert cfg.sde_window_size == 2
    assert cfg.sde_noise_scale == 0.0
    assert sde_step_mask(cfg, 5).tolist() == [False, True, True, False, False]


def test_sampling_config_rejects_mixgrpo_window_past_schedule():
    with pytest.raises(ValueError, match="SDE window exceeds"):
        SamplingConfig.from_mapping({
            "trajectory": "mixed_ode_sde",
            "num_steps": 5,
            "sde_window_start": 4,
            "sde_window_size": 2,
        })


def test_sde_step_mask_maps_existing_trajectories():
    ode = SamplingConfig(num_steps=3, trajectory="ode")
    sde = SamplingConfig(num_steps=3, trajectory="sde_reflow")

    assert sde_step_mask(ode, 3).tolist() == [False, False, False]
    assert sde_step_mask(sde, 3).tolist() == [True, True, True]


def test_sampler_restores_original_batch_timestep_after_sampling():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=2))
    batch = _batch()
    original_timesteps = torch.tensor([123.0])
    batch.timesteps = original_timesteps

    sampler.sample(batch=batch, model=model, generator=torch.Generator().manual_seed(0))

    assert batch.timesteps is original_timesteps


def test_euler_sampler_does_not_renoise_between_steps():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=4))

    sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert model.add_noise_calls == 0
    assert model.timestep_shapes == [(2,), (2,), (2,), (2,)]


def test_sde_reflow_sampler_renoises_between_steps():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=4, trajectory="sde_reflow"))

    sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert model.add_noise_calls == 3


def test_sde_reflow_sampler_scales_renoise():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=4, trajectory="sde_reflow", sde_noise_scale=0.0))

    sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert model.add_noise_calls == 3
    assert all(torch.count_nonzero(noise) == 0 for noise in model.add_noise_inputs)


def test_mixed_sampler_renoises_only_configured_window():
    model = _FakeModel()
    sampler = DiffusionSampler(
        SamplingConfig(
            num_steps=4,
            trajectory="mixed_ode_sde",
            sde_window_start=1,
            sde_window_size=1,
        ))

    sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert model.add_noise_calls == 1
    assert model.predict_noise_calls == 3
    assert model.predict_x0_calls == 1


def test_sampler_with_log_probs_records_only_sde_window():
    model = _FakeModel()
    sampler = DiffusionSampler(
        SamplingConfig(
            num_steps=4,
            trajectory="mixed_ode_sde",
            sde_window_start=1,
            sde_window_size=2,
            sde_noise_scale=0.7,
        ))

    result = sampler.sample_with_log_probs(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert result.trace is not None
    assert result.trace.latents.shape == (2, 2, 1, 3, 4, 4)
    assert result.trace.next_latents.shape == (2, 2, 1, 3, 4, 4)
    assert result.trace.timesteps.shape == (2, 2)
    assert result.trace.log_probs.shape == (2, 2)
    assert result.trace.step_indices.tolist() == [1, 2]
    assert model.predict_noise_calls == 4


def test_diffusion_nft_config_uses_rl_sampler_not_dmd_pipeline():
    config_path = "examples/train/configs/rl/wan/diffusion_nft_pick_clip.yaml"

    cfg = load_run_config(config_path)
    raw_text = open(config_path, encoding="utf-8").read()

    assert cfg.method["_target_"] == "fastvideo.train.methods.rl.diffusion_nft.DiffusionNFTMethod"
    assert cfg.training.optimizer.learning_rate == 3.0e-5
    assert cfg.training.data.num_latent_t == 1
    assert cfg.training.data.num_frames == 1
    assert "sampling_timesteps" not in raw_text
    assert "WanDMDPipeline" not in raw_text
    assert "solver" not in cfg.method["sampling"]
    assert cfg.method["sampling"]["scheduler"] == "flow_match_euler"
    assert cfg.method["sampling"]["trajectory"] == "ode"
    assert cfg.method["sampling"]["flow_shift"] == "inherit"
    assert "deterministic" not in cfg.method["sampling"]
    assert "noise_level" not in cfg.method["sampling"]
    assert cfg.method["validation"]["every_steps"] == 10
    assert cfg.method["validation"]["num_steps"] == 40
    assert cfg.method["validation"]["num_prompts"] == 16
    assert cfg.method["validation"]["log_samples"] is True


def test_mixgrpo_config_uses_sde_window_and_grpo_method():
    config_path = "examples/train/configs/rl/wan/mixgrpo_pick_clip.yaml"

    cfg = load_run_config(config_path)

    assert cfg.method["_target_"] == "fastvideo.train.methods.rl.mix_grpo.MixGRPOMethod"
    assert cfg.method["sampling"]["scheduler"] == "flow_match_euler"
    assert cfg.method["sampling"]["trajectory"] == "mixed_ode_sde"
    assert cfg.method["sampling"]["sde_window_start"] == 5
    assert cfg.method["sampling"]["sde_window_size"] == 10
    assert cfg.method["sampling"]["sde_noise_scale"] == 0.7
    assert cfg.method["clip_range"] == 0.0001
    assert cfg.method["weight_advantages"] is False
    assert cfg.method["prompt_refinement"]["enabled"] is False
    assert cfg.method["prompt_refinement"]["refined_prompt_key"] == "refined_prompt"


def test_validation_shard_indices_are_stable_and_padded():
    rank0 = validation_shard_indices(5, rank=0, world_size=2)
    rank1 = validation_shard_indices(5, rank=1, world_size=2)

    assert rank0 == [(0, True), (2, True), (4, True)]
    assert rank1 == [(1, True), (3, True), (0, False)]


def test_distributed_k_repeat_indices_repeats_prompts_globally():
    rank0 = distributed_k_repeat_indices(
        dataset_length=100,
        batch_size=6,
        repeats_per_prompt=24,
        world_size=4,
        rank=0,
        seed=123,
    )
    all_indices = []
    for rank in range(4):
        sample = distributed_k_repeat_indices(
            dataset_length=100,
            batch_size=6,
            repeats_per_prompt=24,
            world_size=4,
            rank=rank,
            seed=123,
        )
        all_indices.extend(sample.local_indices)

    assert rank0.unique_prompt_count == 1
    assert len(all_indices) == 24
    assert len(set(all_indices)) == 1


def test_validation_caption_puts_rewards_first():
    caption = validation_caption(
        "a small blue cube",
        {
            "avg": 0.75,
            "pickscore": 0.5,
        },
    )

    assert caption.startswith("avg: 0.7500 | pickscore: 0.5000 | ")
    assert caption.endswith("a small blue cube")


def test_media_to_video_array_treats_frame_as_single_frame_video():
    frame = torch.ones(3, 4, 5)

    video = media_to_video_array(frame)

    assert video.shape == (1, 3, 4, 5)
    assert video.dtype.name == "uint8"


def test_media_to_video_array_preserves_video_frames():
    media = torch.ones(3, 2, 4, 5)

    video = media_to_video_array(media)

    assert video.shape == (2, 3, 4, 5)
