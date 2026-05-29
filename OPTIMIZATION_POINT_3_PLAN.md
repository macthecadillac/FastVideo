# Point 3 Plan: Optimize Attention Kernels and Sequence-Parallel Communication

This expands point 3 from `OPTIMIZATION.md` against the codebase as it currently
stands. The plan focuses on repeated video DiT hot paths: every denoising step,
every transformer block, and every self-attention layer.

## Handoff State

Last updated: 2026-05-29 after clarifying numerical parity coverage.

Branch state:

- Current branch: `attn-sp-comm-vsa-compile-optimizations`.
  The `attn-hot-path` branch name was later reused from `main`; the completed
  point-3 optimization work lives on this more descriptive branch.
- Pushed optimization commits:
  - `035eb7e2` `[perf]: add attention SP benchmark harness`
  - `12a70f4a` `[perf]: reduce SP all-to-all layout traffic`
  - `18021fe7` `[perf]: add guarded async replicated gather`
  - `b3f6a3c1` `[perf]: expose VSA tile selection`
  - `13b0d07d` `[perf]: compile dense single-rank attention`
- Known unrelated workspace state remains outside these commits:
  `fastvideo/tests/modal/launch_l40s_job.py` is staged from earlier workspace
  state, and several root/local artifacts remain untracked. Do not include or
  revert them unless the user explicitly asks.

Completed milestones:

- Stage 0 added `tests/local_tests/benchmark_attention_sp.py` and made the
  benchmark emit JSON summaries for communication, attention, and sparse
  metadata cases. Modal L40S:2 baselines were recorded for communication,
  metadata, and dense FlashAttention.
- Stage 1 rewrote sequence-parallel all-to-all layout packing. The
  heads-to-sequence path avoids `split(...); cat(...)`; the sequence-to-heads
  path uses direct packing only for contiguous inputs and keeps the older
  transpose-first path for non-contiguous attention outputs. Modal benchmarks
  improved heads-to-sequence all-to-all from 3.699 ms to 1.032 ms and
  sequence-to-heads from 1.110 ms to 0.399 ms, with dense attention flat.
- Stage 2 added an opt-in inference-only async SP all-gather handle for
  replicated-token attention outputs behind
  `FASTVIDEO_ASYNC_REPLICATED_GATHER=1`. The default remains synchronous.
  Correctness and existing SP gradient parity passed on Modal. The replicated
  attention microbenchmark improved from 5.590 ms sync to 5.364 ms async when
  the wait happens before the following all-to-all.
- Stage 3 made VSA tile size explicit in metadata, added
  `FASTVIDEO_VSA_TILE_SIZE`, kept `(4, 4, 4)` as the default, and exposed
  `auto` selection for the `(4, 8, 8)` BSHD path only when
  `video_sparse_attn_bshd` is installed. It also removed generic `BSA_ATTN`
  from `DenoisingStage`'s allowlist because that stage does not build BSA
  metadata; LongCat's dedicated BSA path is unchanged.
- Stage 4 first slice moved the compiler boundary inside generic
  `DistributedAttention`: dense SP=1 calls with no replicated tokens and default
  QKV layout hooks now skip QKV concat/chunk plus no-op all-to-all and can be
  captured by `torch.compile(..., fullgraph=True)`. The multi-rank and
  sparse-layout fallback remains behind `@torch.compiler.disable`.

Validation summary:

- Local syntax checks were run with `python -m py_compile` for touched Python
  files in each stage.
- Pre-commit was run on each committed slice with the repository's configured
  hooks.
- Modal L40S tests run during the completed stages:
  - `pytest fastvideo/tests/distributed/test_all_to_all_4d_layout.py -sv`
  - `pytest fastvideo/tests/distributed/test_async_all_gather.py -sv`
  - `pytest fastvideo/tests/attention/test_sparse_attention_wiring.py -sv`
  - `pytest fastvideo/tests/attention/test_distributed_attention_compile.py -sv`
  - `pytest fastvideo/tests/distributed/test_sp_wan.py fastvideo/tests/distributed/test_sp_ltx2.py fastvideo/tests/distributed/test_sp_hunyuanvideo.py -sv`
- Modal benchmark runs are documented in the per-stage handoff notes below,
  including app IDs and headline timings.

Numerical parity status:

- Near bit-by-bit branch-vs-main parity was not established for point 3. The
  completed validation did not run a full end-to-end generated latent or video
  comparison between `main` and this branch.
- The correct claim for this branch is targeted component and tiny-model parity
  within tight floating-point tolerances:
  - Stage 1 all-to-all rewrites passed deterministic layout tests, round-trip
    checks, and backward checks with `torch.testing.assert_close`. Existing
    Wan, LTX2, and HunyuanVideo SP gradient parity tests also passed on Modal;
    those tests compare single-rank and SP gradients with
    `rtol=1e-4, atol=1e-5`.
  - Stage 2 async replicated gather passed sync-vs-async all-gather checks and
    replicated-token attention output parity with `rtol=1e-5, atol=1e-5`.
    This path remains opt-in via `FASTVIDEO_ASYNC_REPLICATED_GATHER=1`, so the
    default inference/training behavior is still synchronous.
  - Stage 3 sparse-attention wiring validated tile-size selection and fallback
    behavior, but did not validate numerical parity for the fast
    `video_sparse_attn_bshd` path because that kernel was not installed in the
    Modal L40S image. No production speedup or output-parity claim is made for
    the unavailable fast sparse kernel.
  - Stage 4 dense SP=1 direct path matched `LocalAttention` with
    `rtol=1e-5, atol=1e-5`, and
    `torch.compile(..., backend="eager", fullgraph=True)` matched eager output
    with the same tolerance. Existing Wan, LTX2, and HunyuanVideo SP gradient
    parity tests passed again after this change.
- Bitwise equality is not a realistic or claimed target for these attention and
  distributed changes. Kernel call shape, operation ordering, distributed
  collectives, and compile boundaries can all change floating-point rounding
  while preserving numerically equivalent results. The intended gate is tight
  `allclose` parity plus model-level latent or SSIM checks for paths that can
  change generated outputs.
- Recommended follow-up before making broader claims or enabling optional sparse
  paths by default:
  - Run branch-vs-main Wan 1.3B latent-output parity on Modal with fixed seed,
    prompt, dimensions, and scheduler settings. Record `max_abs`, `mean_abs`,
    and `torch.allclose` results.
  - Run the same comparison for SP=1 and SP=2 attention profiles under
    `FLASH_ATTN`; keep `TORCH_SDPA` coverage for deterministic compile tests.
  - If `video_sparse_attn_bshd` is installed and enabled, run sparse-kernel
    latent parity plus decoded-frame SSIM/visual review before changing any
    default sparse-attention behavior.

Next resume point:

- Continue Phase 4 with model-level compile coverage: add a tiny Wan-like
  eager-vs-compiled path that exercises the repository's `_compile_conditions`
  / `enable_torch_compile` plumbing, now that the attention wrapper itself has
  a fullgraph-covered SP=1 dense path.
- Add an explicit branch-vs-main latent parity harness if review requires a
  full before/after output-equivalence claim.
- Keep distributed collectives and sparse-layout hooks out of the compiled
  path until there is a dedicated test proving Dynamo behavior for those
  branches.
- Continue updating this `Handoff State` section after every major milestone
  before compacting or ending a long session.

## Performance Gains by Stage

All numbers below come from the Modal L40S benchmark and validation runs
recorded in the stage handoff notes. They are microbenchmark results, not a
full end-to-end serving latency claim. The safe interpretation is per-hot-path
speedup under the synthetic shapes in `tests/local_tests/benchmark_attention_sp.py`.

### Stage 0: Baseline Harness

Stage 0 did not change runtime behavior. It added the benchmark harness and
recorded the reference numbers used by later stages.

Baseline measurements:

- SP=2 heads-to-sequence all-to-all:
  3.699 ms max rank average.
- SP=2 sequence-to-heads all-to-all:
  1.110 ms max rank average.
- SP=2 replicated-token all-gather:
  0.160 ms max rank average.
- SP=2 dense FlashAttention wrapper:
  1.780 ms max rank average.
- Sparse metadata construction:
  VSA 15.95 us, BSA 3.30 us, VMoBA 2.30 us.

Performance result:

- No speedup claimed. The gain was observability: the repo now has a JSON
  benchmark that isolates communication, attention wrapper, replicated-token
  gather, and sparse metadata paths without model weights.

### Stage 1: Sequence-Parallel All-To-All Layout Traffic

Measured gains:

- Heads-to-sequence all-to-all improved from 3.699 ms to 1.032 ms.
  This is a 72.1% latency reduction, or 3.58x faster.
- Sequence-to-heads all-to-all improved from 1.110 ms to 0.399 ms.
  This is a 64.1% latency reduction, or 2.78x faster.
- The combined two-direction communication microbenchmark improved from
  4.809 ms to 1.431 ms. This is a 70.2% reduction, or 3.36x faster.
- Dense SP=2 FlashAttention wrapper timing stayed flat: 1.780 ms before and
  1.772 ms after, a 0.4% difference within noise for the end-to-end wrapper
  benchmark.
- Reported peak allocation for the communication benchmark stayed unchanged.

What caused the gain:

- The heads-to-sequence path stopped materializing `split(...); cat(...)`
  chunks after the receive buffer and instead reshaped/permuted directly into
  attention layout.
- The sequence-to-heads path now uses direct contiguous packing when the input
  is contiguous, while preserving the older transpose-first packing for
  non-contiguous attention outputs. That fallback matters: an always-direct
  attempt regressed dense attention from about 1.78 ms to 6.21 ms.

Performance interpretation:

- Stage 1 produced the largest isolated communication gain so far.
- The dense attention wrapper did not get faster in the synthetic SP=2 run
  because the measured attention kernel and other wrapper work dominate that
  particular end-to-end microbenchmark. The isolated all-to-all win is still
  substantial and should matter most in profiles where communication/layout
  traffic is a visible share of block time.

### Stage 2: Async Replicated-Token Gather

Measured gains:

- Replicated-token attention with the synchronous gather measured 5.590 ms.
- The final async placement measured 5.364 ms.
- This is a 4.0% latency reduction, or 1.04x faster, for the replicated-token
  attention microbenchmark.
- A first async attempt measured 6.517 ms, a 16.6% regression versus sync. That
  version waited after the following all-to-all and was not kept.

What caused the gain:

- The accepted implementation starts the replicated-output all-gather, performs
  independent local postprocess/padding work, waits for the gather, and only
  then starts the following main-output all-to-all.
- The path is inference-only and opt-in via
  `FASTVIDEO_ASYNC_REPLICATED_GATHER=1`; training and the default inference path
  remain synchronous.

Performance interpretation:

- The measured speedup is modest but real on the synthetic replicated-token
  shape.
- The result is sensitive to wait placement. Starting an async collective is not
  enough; the code must have useful local work between launch and wait, and it
  must avoid overlapping independent NCCL collectives in a way that serializes
  worse than the sync path.
- Because the win is narrow and shape-dependent, Stage 2 intentionally did not
  make async replicated gathers the default.

### Stage 3: Sparse-Attention Fast-Path Selection

Measured results:

- The Modal L40S image did not include `video_sparse_attn_bshd`, so
  `FASTVIDEO_VSA_TILE_SIZE=auto` and explicit 256-token requests correctly fell
  back to the legacy `(4, 4, 4)` tile.
- Stage 0 VSA metadata baseline was 15.95 us.
- Stage 3 default VSA metadata run measured 23.900 us with `(4, 4, 4)`.
- Stage 3 auto fallback metadata run measured 14.367 us, still with
  `(4, 4, 4)` after fallback.

What changed:

- VSA tile size is now explicit in metadata and configurable by
  `FASTVIDEO_VSA_TILE_SIZE` or the benchmark's `--vsa-tile-size`.
- The 256-token BSHD path is now reachable when `video_sparse_attn_bshd` is
  installed, while the default remains the legacy 64-token tile.
- Generic `BSA_ATTN` was removed from `DenoisingStage`'s allowlist because that
  stage does not build BSA metadata.

Performance interpretation:

- No production speedup is claimed for Stage 3 on the current L40S image. The
  fast BSHD kernel dependency was unavailable, so the benchmark could only
  validate fallback behavior.
- The metadata numbers should be treated as validation and run-to-run context,
  not as a claimed VSA speedup. The default run was slower than Stage 0, while
  the auto fallback run was slightly faster; both used the same fallback tile.
- The performance value of Stage 3 is enabling and safety work: it makes the
  fast sparse path selectable and inspectable on images where the kernel exists,
  and it prevents the generic denoising stage from selecting unsupported BSA
  wiring.

### Stage 4: Dense SP=1 Compile-Visible Direct Path

Measured gains:

- Dense SP=1 FlashAttention wrapper before the patch measured 0.231014 ms.
- Dense SP=1 FlashAttention wrapper after the patch measured 0.202752 ms.
- This is a 12.2% latency reduction, or 1.14x faster.
- Peak allocated memory dropped from 113,246,720 bytes to 88,080,896 bytes.
  This is a 22.2% reduction, or 24 MiB less peak allocation in the benchmark.

What caused the gain:

- Dense SP=1 calls with no replicated tokens and default QKV layout hooks now
  skip QKV `cat`, no-op single-rank all-to-all, QKV `chunk`, and final no-op
  all-to-all.
- `DistributedAttention.forward` is compile-visible for that dense SP=1 branch,
  and the distributed/sparse fallback remains behind `@torch.compiler.disable`.

Performance interpretation:

- Stage 4 is the clearest direct attention-wrapper win so far.
- The gain applies only to dense SP=1 calls that do not use replicated tokens
  and do not require backend-specific QKV layout hooks.
- The full model-level compile benefit is not measured yet. The next Stage 4
  slice should add a tiny Wan-like eager-vs-compiled model test that exercises
  `_compile_conditions` or `enable_torch_compile` end to end.

### Cumulative Readout

- Biggest isolated communication win: Stage 1, with the two all-to-all
  directions together dropping from 4.809 ms to 1.431 ms.
- Biggest direct attention-wrapper win: Stage 4, with dense SP=1 FlashAttention
  dropping from 0.231014 ms to 0.202752 ms and peak allocation dropping by
  24 MiB.
- Narrow opt-in overlap win: Stage 2, with replicated-token attention dropping
  from 5.590 ms to 5.364 ms when async gather is explicitly enabled.
- Enabling/safety stage without measured speedup on this image: Stage 3, because
  the BSHD sparse kernel was not installed on the Modal L40S image.

## Current State

The central generic path is `fastvideo/attention/layer.py::DistributedAttention`.
It now has a compile-visible `forward` that dispatches dense SP=1 calls with no
replicated tokens and default QKV layout hooks into `_forward_single_rank_dense`.
That direct path trims SP padding, applies optional RoPE to Q/K, calls the
selected attention backend without QKV concat/chunk or all-to-all, then pads the
output back. Multi-rank, replicated-token, and sparse-layout paths still route
through `_forward_sequence_parallel`, which remains marked
`@torch.compiler.disable`.

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

Stage 2 handoff:

- Status: implemented and remotely validated on Modal L40S:2.
- Current edits:
  `fastvideo/distributed/device_communicators/base_device_communicator.py`,
  `fastvideo/distributed/parallel_state.py`,
  `fastvideo/distributed/communication_op.py`, `fastvideo/envs.py`,
  `fastvideo/attention/layer.py`, `fastvideo/models/dits/ltx2.py`,
  `fastvideo/tests/distributed/test_async_all_gather.py`, and
  `tests/local_tests/benchmark_attention_sp.py`.
- Added `FASTVIDEO_ASYNC_REPLICATED_GATHER=1` as an opt-in guard. The default
  remains synchronous.
- Added an inference-only async SP all-gather handle. It preserves the same
  logical output layout as the existing synchronous all-gather and rejects
  autograd-tracked tensors while grad mode is enabled.
- Wired the guarded path into generic `DistributedAttention` and the LTX2
  distributed attention wrapper for replicated-token outputs only.
- The async gather is started before local output postprocess/padding/main
  all-to-all work and is waited before returning the replicated output.
- Added distributed coverage for async-vs-sync all-gather, the grad-enabled
  rejection path, and replicated-token attention output parity.
- Extended the attention/SP benchmark with `--include-replicated-attention` so
  Stage 2 can compare sync and async replicated-token gather timing.
- Local validation completed:
  `python -m py_compile fastvideo/distributed/device_communicators/base_device_communicator.py fastvideo/distributed/parallel_state.py fastvideo/distributed/communication_op.py fastvideo/envs.py fastvideo/attention/layer.py fastvideo/models/dits/ltx2.py fastvideo/tests/distributed/test_async_all_gather.py tests/local_tests/benchmark_attention_sp.py`.
- Pre-commit completed:
  `pre-commit run --files OPTIMIZATION_POINT_3_PLAN.md fastvideo/distributed/device_communicators/base_device_communicator.py fastvideo/distributed/parallel_state.py fastvideo/distributed/communication_op.py fastvideo/envs.py fastvideo/attention/layer.py fastvideo/models/dits/ltx2.py fastvideo/tests/distributed/test_async_all_gather.py tests/local_tests/benchmark_attention_sp.py`.
- Modal async correctness completed:
  `pytest fastvideo/tests/distributed/test_async_all_gather.py -sv` on app
  `ap-xot47iiqfvKlHfWiXBhAGK`.
- Existing SP parity completed:
  `pytest fastvideo/tests/distributed/test_sp_wan.py fastvideo/tests/distributed/test_sp_ltx2.py fastvideo/tests/distributed/test_sp_hunyuanvideo.py -sv`
  on app `ap-KGkOgmUUbv5MRTXfce9AgU`; all 3 tests passed.
- Replicated attention sync benchmark completed on app
  `ap-qoW1gzihCm7Oket3rIQBE7`: replicated attention max rank average
  5.590 ms.
- Replicated attention async benchmark completed on app
  `ap-3c6czFkXjIMyykZWhk2S0T`: replicated attention max rank average
  5.364 ms with the wait placed before the following all-to-all.
- A first async attempt waited after the following all-to-all and benchmarked
  slower at 6.517 ms on app `ap-wVVHILeHM4F4cufxDsKEPg`; the final code waits
  before starting the next collective.
- Default remains synchronous. The async path is only enabled by
  `FASTVIDEO_ASYNC_REPLICATED_GATHER=1`, because this is a narrow benchmark win
  and should be confirmed on real replicated-token model shapes before becoming
  a preset/default.
- Resume point: Stage 2 is ready to commit. Stage 3 should focus on sparse
  attention fast-path selection and generic BSA stage wiring.

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

Stage 3 handoff:

- Status: implemented and remotely validated on Modal L40S:1.
- Current edits:
  `fastvideo/attention/backends/video_sparse_attn.py`, `fastvideo/envs.py`,
  `fastvideo/pipelines/stages/denoising.py`,
  `fastvideo/tests/attention/test_sparse_attention_wiring.py`, and
  `tests/local_tests/benchmark_attention_sp.py`.
- VSA now records `tile_size` in metadata and no longer relies on the
  module-level `VSA_TILE_SIZE` constant for metadata construction, tiling, or
  kernel dispatch.
- The default VSA tile remains `(4, 4, 4)`. `FASTVIDEO_VSA_TILE_SIZE=auto`
  selects `(4, 8, 8)` only when `video_sparse_attn_bshd` is importable;
  explicit `VSA_tile_size`/`--vsa-tile-size` overrides are also supported.
- 256-token VSA tiles still route through the BSHD kernel path; unsupported
  explicit 256-token requests fall back to `(4, 4, 4)` with a warning.
- Generic `BSA_ATTN` was removed from the denoising-stage backend allowlist
  because that generic stage does not build BSA metadata. LongCat's dedicated
  BSA path is separate and unchanged.
- The benchmark metadata suite now reports the VSA tile size and supports
  `--vsa-tile-size`.
- Added sparse wiring tests for VSA tile selection and for ensuring generic
  denoising does not select `BSA_ATTN` from the global env override.
- Local validation completed:
  `python -m py_compile fastvideo/attention/backends/video_sparse_attn.py fastvideo/envs.py fastvideo/pipelines/stages/denoising.py fastvideo/tests/attention/test_sparse_attention_wiring.py tests/local_tests/benchmark_attention_sp.py`.
- Pre-commit completed:
  `pre-commit run --files OPTIMIZATION_POINT_3_PLAN.md fastvideo/attention/backends/video_sparse_attn.py fastvideo/envs.py fastvideo/pipelines/stages/denoising.py fastvideo/tests/attention/test_sparse_attention_wiring.py tests/local_tests/benchmark_attention_sp.py`.
- Modal sparse wiring completed:
  `pytest fastvideo/tests/attention/test_sparse_attention_wiring.py -sv` on
  app `ap-P4azSWjK7pjIiO1CASEDb7`; both tests passed.
- The Modal L40S image does not include `video_sparse_attn_bshd`, so the
  `auto` and explicit 256-token test paths correctly fall back to `(4, 4, 4)`.
- Default metadata benchmark completed on app `ap-KTP2XnvFDTXAQ2954WHyR7`:
  VSA metadata max rank average 23.900 us with tile size `(4, 4, 4)`.
- Auto metadata benchmark completed on app `ap-pRlKQdPGHp0LVYQjKtRviv`:
  VSA metadata max rank average 14.367 us with tile size `(4, 4, 4)` after
  fallback.
- Resume point: Stage 3 is ready to commit. A future Blackwell or image with
  `video_sparse_attn_bshd` installed should rerun the same metadata benchmark
  with `--vsa-tile-size auto` and then a VSA forward benchmark before making
  the 256-token tile default.

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

Stage 4 first-slice handoff:

- Status: implemented, committed as `13b0d07d`, and remotely validated on Modal
  L40S:1/L40S:2.
- Current edits: `fastvideo/attention/layer.py` and
  `fastvideo/tests/attention/test_distributed_attention_compile.py`.
- `DistributedAttention.forward` is no longer globally compiler-disabled.
  Dense SP=1 calls use `_forward_single_rank_dense` when there are no
  replicated tokens and the selected backend uses the default QKV
  preprocess/postprocess hooks. This keeps sparse/BSA/VSA layout hooks on the
  existing fallback.
- `_forward_single_rank_dense` avoids the previous QKV `cat`, no-op
  single-rank all-to-all, QKV `chunk`, and final no-op all-to-all. It preserves
  the prior trim/RoPE/pad behavior.
- `_forward_sequence_parallel` contains the previous distributed implementation
  and remains `@torch.compiler.disable`, so SP>1 collectives are not newly
  exposed to Dynamo.
- The new compile test runs under `torchrun --nproc_per_node=1`, initializes
  FastVideo distributed state with `sp_size=1`, verifies the direct path does
  not call `sequence_model_parallel_all_to_all_4D`, compares against
  `LocalAttention` with RoPE and padding, and verifies
  `torch.compile(distributed_attention.forward, backend="eager",
  fullgraph=True)` matches eager output.
- Local validation completed:
  `python -m py_compile fastvideo/attention/layer.py fastvideo/tests/attention/test_distributed_attention_compile.py`.
- Pre-commit completed:
  `pre-commit run --files fastvideo/attention/layer.py fastvideo/tests/attention/test_distributed_attention_compile.py`.
- Modal compile/correctness test completed on app
  `ap-ARuc2XCOIktTlFMhciiBfB`:
  `pytest fastvideo/tests/attention/test_distributed_attention_compile.py -sv`
  passed. That run used `--install-extra dev`; the test forces `TORCH_SDPA`.
- FlashAttention SP=1 benchmark before the patch completed on app
  `ap-zyQ03eA9KX7VbNrBVwEC5B` with `--install-extra none`:
  `distributed_attention_dense` max rank average 0.231014 ms and peak
  allocated memory 113,246,720 bytes.
- FlashAttention SP=1 benchmark after the patch completed on app
  `ap-go5Ae0Rz3vC9xFHRxpEPzA` with the same command and
  `--install-extra none`: max rank average 0.202752 ms and peak allocated
  memory 88,080,896 bytes.
- A benchmark attempt on app `ap-15usZhJT9bCM6sTS3veFRZ` is intentionally not
  used for comparison: installing `.[dev]` upgraded the environment to a torch
  build without `flash_attn`, so the benchmark fell back to SDPA and rejected
  the requested `FLASH_ATTN` backend.
- Existing SP gradient parity completed on app `ap-VaLs2XofuIeBEMaPjPjAR1`
  with the patch applied:
  `pytest fastvideo/tests/distributed/test_sp_wan.py fastvideo/tests/distributed/test_sp_ltx2.py fastvideo/tests/distributed/test_sp_hunyuanvideo.py -sv`
  passed 3 tests in 78.90 seconds.
- Resume point: the next Stage 4 slice should add model-level
  `enable_torch_compile` coverage for a tiny Wan-like model or pipeline compile
  path. Do not remove the fallback compiler disable around SP>1 collectives
  until a dedicated distributed compile test exists.

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
