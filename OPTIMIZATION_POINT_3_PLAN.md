# Point 3 Plan: Optimize Attention Kernels and Sequence-Parallel Communication

This expands point 3 from `OPTIMIZATION.md` against the codebase as it currently
stands. The plan focuses on repeated video DiT hot paths: every denoising step,
every transformer block, and every self-attention layer.

## Current State

The central generic path is `fastvideo/attention/layer.py::DistributedAttention`.
It is currently marked `@torch.compiler.disable`, concatenates Q/K/V, runs a
sequence-parallel all-to-all, optionally applies RoPE, optionally appends
replicated tokens, chunks Q/K/V again, runs the selected attention backend, and
then runs a second all-to-all.

The sequence-parallel 4D all-to-all implementation lives in
`fastvideo/distributed/device_communicators/base_device_communicator.py`. It
does multiple layout conversions around `dist.all_to_all_single`, including
`transpose(...).contiguous()` and a `split` plus `cat` in the heads-to-sequence
direction.

Sparse attention support is split across multiple paths:

- VSA is wired through `DistributedAttention_VSA` and LTX2's specialized wrapper.
- VSA has a BSHD fast path for 256-token tiles, but the global tile size is
  currently `(4, 4, 4)`, so the default 64-token path still transposes to BHSD.
- VMoBA metadata is built in `DenoisingStage`.
- `BSA_ATTN` is listed in `DenoisingStage`'s attention allowlist, but generic
  BSA metadata is not built there; LongCat uses its own third-party BSA path.

The existing performance test reports end-to-end and stage-level timings, but it
does not isolate attention communication, layout conversions, sparse metadata
construction, or per-backend kernel time.

## Goals

1. Reduce unnecessary tensor materialization around Q/K/V and all-to-all.
2. Overlap collectives only when there is real independent work to hide them
   behind.
3. Make sparse-attention workloads select fast kernel paths when shape and
   hardware constraints allow it.
4. Remove compiler graph breaks where doing so is safe and measurable.
5. Keep dense attention, sparse attention, SP correctness, and training
   gradients covered by targeted tests.

## Non-Goals

- Do not rewrite model architectures to use different attention semantics.
- Do not change checkpoint state-dict surfaces.
- Do not migrate training pipelines between `fastvideo/training/` and
  `fastvideo/train/`.
- Do not make sparse backends default for all models without model-specific
  quality and latency evidence.

## Phase 0: Benchmark Harness and Baselines

Add microbenchmarks before changing hot-path behavior.

Suggested coverage:

- `sequence_model_parallel_all_to_all_4D` for both supported directions:
  `scatter_dim=2,gather_dim=1` and `scatter_dim=1,gather_dim=2`.
- `sequence_model_parallel_all_gather` for replicated-token and LTX2
  cross-modal shapes.
- `DistributedAttention.forward` with Wan-like dense shapes.
- `DistributedAttention_VSA.forward` with Wan/LTX2 VSA-like shapes.
- Sparse backend metadata build time for VSA, VMoBA, and generic BSA.

Metrics:

- Wall time via CUDA events, not production `.item()` timing.
- Allocated bytes and peak reserved memory.
- Tensor contiguity/layout before and after the communication helpers.
- NCCL operation count per layer.

Likely locations:

- `tests/local_tests/benchmark_attention_sp.py` for exploratory local runs.
- `fastvideo/tests/performance/` only after the benchmark shape and output format
  are stable enough for CI-style reporting.

Acceptance:

- Baselines include at least Wan 1.3B-style SP=2 shapes and a small synthetic
  shape that runs quickly.
- Benchmarks can be run with `torchrun --nproc_per_node=2`.
- Output is JSON so later PRs can compare before/after without manual parsing.

Stage 0 handoff:

- Status: implemented and remotely validated on Modal L40S:2.
- Current artifact: `tests/local_tests/benchmark_attention_sp.py`.
- The benchmark is intentionally synthetic and uses the real FastVideo
  distributed and attention wrappers, so it can run without model weights.
- FastVideo imports are lazy so argument parsing works on CPU-only hosts; CUDA
  is still required for actual benchmark execution.
- Local validation completed:
  `python -m py_compile tests/local_tests/benchmark_attention_sp.py` and
  `python tests/local_tests/benchmark_attention_sp.py --help`.
- Modal validation completed:
  `torchrun --standalone --nnodes=1 --nproc_per_node=2 tests/local_tests/benchmark_attention_sp.py --suites communication,metadata --iterations 5 --warmup 2 --metadata-iterations 10 --metadata-warmup 2 --output /tmp/attention_sp_baseline.json`
- Communication/metadata baseline from Modal app
  `ap-nYL4j1tZVWll1jlJicGz5p`: heads-to-sequence all-to-all max rank average
  3.699 ms, sequence-to-heads all-to-all 1.110 ms, replicated-token all-gather
  0.160 ms, VSA metadata 15.95 us, BSA metadata 3.30 us, VMoBA metadata
  2.30 us.
- FlashAttention dense benchmark completed with:
  `torchrun --standalone --nnodes=1 --nproc_per_node=2 tests/local_tests/benchmark_attention_sp.py --suites attention --attention-backend FLASH_ATTN --iterations 5 --warmup 2 --output /tmp/attention_sp_attention_flash.json`
- Dense attention baseline from Modal app `ap-ASNgzZ49xV4SwUlT7az3Az`:
  distributed attention max rank average 1.780 ms.
- Resume point: Stage 1 can now compare layout changes against these baseline
  commands and numbers.

## Phase 1: Reduce Layout Traffic in Sequence-Parallel All-To-All

Start with implementation-preserving rewrites of the current 4D all-to-all.

Work items:

- Replace the heads-to-sequence `output.split(...); torch.cat(...)` with
  reshape/view/movedim logic that avoids creating a Python tuple of chunks and
  a new concatenation allocation.
- Audit whether both final `.contiguous()` calls are required by downstream
  kernels. Keep them where the next op requires contiguous memory, remove or
  defer them where a view is sufficient.
- Add explicit shape assertions for divisibility by `world_size` before any
  reshape to catch invalid layouts early.
- Add an SP=1 fast path in `DistributedAttention` that avoids QKV `cat`/`chunk`
  and directly calls `attn_impl.forward(q, k, v, metadata)` when no
  replicated-token handling or distributed redistribution is needed.

Tests:

- Existing SP gradient parity tests:
  - `fastvideo/tests/distributed/test_sp_wan.py`
  - `fastvideo/tests/distributed/test_sp_ltx2.py`
  - `fastvideo/tests/distributed/test_sp_hunyuanvideo.py`
- New unit-level all-to-all layout tests with deterministic tensors and
  round-trip checks.
- Benchmark comparison from Phase 0.

Acceptance:

- SP output and gradient parity remain within existing tolerances.
- No increase in peak memory for benchmarked all-to-all shapes.
- At least one measured reduction in layout conversion time or allocation count
  before moving to higher-risk async work.

Stage 1 handoff:

- Status: implemented and remotely validated on Modal L40S:2.
- Current edits:
  `fastvideo/distributed/device_communicators/base_device_communicator.py` and
  `fastvideo/tests/distributed/test_all_to_all_4d_layout.py`.
- The heads-to-sequence all-to-all no longer materializes Python split chunks
  before concatenation. It reshapes the all-to-all receive buffer as
  `[source_rank, local_head, local_seq, batch, head_dim]` and directly emits the
  `[batch, full_seq, local_head, head_dim]` layout.
- The sequence-to-heads all-to-all prepares the send buffer directly from the
  original `[batch, full_seq, local_head, head_dim]` layout when that input is
  contiguous.
- Non-contiguous sequence-to-heads input keeps the older transpose-first
  packing path. A first attempt to always use the direct packing path made the
  dense FlashAttention benchmark regress from roughly 1.78 ms to 6.21 ms, so
  the final patch keeps the optimized path only where the benchmark supports
  it.
- Both directions now fail early if the scattered dimension is not divisible by
  SP world size.
- Deterministic layout coverage was added for both directions plus a
  heads-to-sequence/sequence-to-heads round trip and backward pass. The
  sequence-to-heads test covers both contiguous and non-contiguous inputs.
- Local validation completed:
  `python -m py_compile fastvideo/distributed/device_communicators/base_device_communicator.py fastvideo/tests/distributed/test_all_to_all_4d_layout.py`.
- Local `pytest fastvideo/tests/distributed/test_all_to_all_4d_layout.py -q`
  is blocked in this CPU-only sandbox because `fastvideo/tests/conftest.py`
  imports `fastvideo_kernel`, which initializes Triton without a CUDA driver.
- Modal validation completed:
  `pytest fastvideo/tests/distributed/test_all_to_all_4d_layout.py -sv`
  on app `ap-ucaalOc51Qqu6hzcWj8iEX`.
- Existing SP parity completed:
  `pytest fastvideo/tests/distributed/test_sp_wan.py fastvideo/tests/distributed/test_sp_ltx2.py fastvideo/tests/distributed/test_sp_hunyuanvideo.py -sv`
  on app `ap-XFEMqVglGkcfPMCy5rB63L`; all 3 tests passed.
- Final communication benchmark completed on app `ap-ZoPvIb0S7qDAnFVlQ8xzBu`.
  Compared with Stage 0, heads-to-sequence all-to-all improved from 3.699 ms
  to 1.032 ms, sequence-to-heads all-to-all improved from 1.110 ms to
  0.399 ms, and reported peak allocation stayed unchanged.
- Final dense FlashAttention benchmark completed on app
  `ap-NTug2obs5Bo4U8RgrLhPKO`: distributed attention max rank average
  1.772 ms, flat to the 1.780 ms Stage 0 baseline.
- Resume point: Stage 1 is ready to commit. Stage 2 should start with
  inference-only async gather experiments, not additional all-to-all rewrites,
  unless a real model trace exposes another layout-specific issue.

## Phase 2: Overlap Independent Collectives

Do not start by making the main attention all-to-alls async. The attention
kernel depends on the first all-to-all, and downstream projection depends on the
second. Begin with collectives that already have independent local work.

Initial candidates:

- Replicated-token output gather in `DistributedAttention.forward`, currently
  called before `postprocess_output`, padding, and the main output all-to-all.
- LTX2 cross-modal gathers:
  - `ax_context = sequence_model_parallel_all_gather(ax_scaled, dim=1)`
  - `ax_full = sequence_model_parallel_all_gather(ax_scaled, dim=1)`
  - `vx_context = sequence_model_parallel_all_gather(vx_scaled, dim=1)`

Work items:

- Add inference-only async all-gather helpers in `fastvideo.distributed` that
  return a small handle object with `wait()`.
- Keep the existing autograd-aware synchronous path for training.
- Use `torch.distributed` async handles first for correctness; consider PyNccl
  stream-based implementations only after measuring the simpler version.
- Ensure every async path has a clear wait point before returned tensors are
  consumed.
- Guard async paths behind a runtime flag until they have stable benchmark data.

Tests:

- Tiny distributed tests that compare sync and async gather outputs.
- Wan and LTX2 SP parity tests under no-grad inference mode.
- Training gradient tests stay on the synchronous path.

Acceptance:

- No deadlocks under `torchrun --nproc_per_node=2`.
- No correctness drift versus sync collectives.
- Measured neutral-or-better latency on the targeted shapes before enabling by
  default.

## Phase 3: Make Sparse-Attention Workloads Hit Fast Paths

VSA and BSA can be faster only when the selected kernel path matches the shape.
The current code leaves some fast paths unreachable or poorly integrated.

Work items:

- Make VSA tile size configurable through metadata or backend config instead of
  the module-level `VSA_TILE_SIZE` constant.
- Add a runtime selector for VSA:
  - Use `(4, 8, 8)` and the BSHD fast path when `video_sparse_attn_bshd` is
    installed, head size is supported, and the latent shape satisfies kernel
    constraints.
  - Fall back to `(4, 4, 4)` when constraints fail.
- Move VSA metadata builder construction out of the per-step inner branch where
  possible. The builder object is stateless today, and only the metadata depends
  on timestep/shape.
- Wire generic `BSA_ATTN` metadata in `DenoisingStage` or remove `BSA_ATTN` from
  that generic allowlist. The current half-wired state can select BSA without
  providing the metadata its backend expects.
- For BSA, reduce Python-loop-heavy paths only after defining the intended
  production path. The pure-PyTorch fallback is useful for tests, but not a
  likely performance target.

Tests:

- VSA metadata tests for 64-token and 256-token tile modes.
- VSA backend smoke test that verifies the BSHD branch is selected when
  available.
- BSA stage wiring test that verifies metadata is present when `BSA_ATTN` is
  selected, or verifies BSA is rejected cleanly if left unsupported.
- SSIM or latent similarity tests for any model where sparse attention becomes
  easier to enable.

Acceptance:

- Sparse backend selection is explicit and inspectable in logs.
- Unsupported sparse shapes fail early with actionable errors, not later in a
  kernel.
- VSA fast-path benchmarks show a real improvement before changing defaults.

## Phase 4: Revisit Graph Breaks Around Attention

The FlashAttention FA2/FA3 default path is already wrapped as a traceable custom
op for inference. The larger graph break is now the distributed wrapper itself.

Work items:

- Remove `@torch.compiler.disable` only from the SP=1 dense path first.
- Keep distributed collectives behind small explicit wrappers so Dynamo sees a
  stable boundary.
- Add compile tests for `enable_torch_compile=True` on a tiny Wan-like model.
- Re-test the documented `torch.compile` caveat before recommending
  `mode="reduce-overhead"` or CUDA graph modes.
- Do not claim training compile support unless custom op backward coverage is
  added for the relevant backend.

Tests:

- `enable_torch_compile=True` eager-vs-compiled output comparison on a tiny
  model.
- Smoke generation with Wan 1.3B-style config after one warmup generation.
- Existing FlashAttention custom-op tests.

Acceptance:

- Compile no longer sees the dense SP=1 wrapper as a graph break.
- No output drift beyond expected floating-point tolerance.
- Warmup cost and steady-state latency are reported separately.

## Phase 5: End-to-End Validation

Run validation in layers, from narrow correctness to user-visible output.

Correctness:

- `pytest fastvideo/tests/distributed/test_sp_wan.py -sv`
- `pytest fastvideo/tests/distributed/test_sp_ltx2.py -sv`
- `pytest fastvideo/tests/distributed/test_sp_hunyuanvideo.py -sv`
- Targeted attention backend unit tests.

Performance:

- Phase 0 microbenchmarks before/after each PR.
- Existing Wan 1.3B 2-GPU performance config in
  `.buildkite/performance-benchmarks/tests/wan-t2v-1.3b.json`.
- Stage timing with `FASTVIDEO_STAGE_LOGGING=1`.

Quality:

- Existing SSIM tests for changed model paths when output numerics can change.
- For sparse-attention default changes, seed or reseed references only after
  manual review of generated videos.

## Suggested PR Order

1. Add attention/SP microbenchmarks and JSON output.
2. Rewrite all-to-all layout handling without semantic changes.
3. Add dense SP=1 direct path in `DistributedAttention`.
4. Add guarded async all-gather for replicated-token or LTX2 cross-modal paths.
5. Make VSA tile-size selection configurable and benchmarked.
6. Resolve generic BSA stage wiring.
7. Remove compile disable for the dense SP=1 path and add compile coverage.

## Risks

- Layout rewrites can silently preserve shape while changing token/head order.
  Use deterministic round-trip tests with non-symmetric values.
- Async collectives can deadlock if one rank takes a different branch. Keep
  branch conditions based on shared metadata, not rank-local tensor contents.
- Sparse backends are numerics-changing optimizations. Treat quality gates as
  mandatory before enabling them by default.
- Training uses custom autograd wrappers for distributed collectives. Do not
  route grad-enabled calls through inference-only async helpers.
