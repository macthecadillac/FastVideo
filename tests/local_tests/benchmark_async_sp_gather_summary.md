# Async Sequence-Parallel Gather Work Summary

Branch: `asynchronous-gather`

Base: `origin/main` at `ba4c02d8`

Implementation commit: `23e1da29`

Validation/artifact commits: `1c34762a`, `bcdeca62`

## Commit Summary

| Commit | Subject | Purpose |
|---|---|---|
| `23e1da29` | `[perf]: async replicated SP gather` | Added a non-autograd asynchronous sequence-parallel all-gather path and used it in no-grad `DistributedAttention` replicated-token output gathering. |
| `1c34762a` | `[test]: cover async SP gather` | Added a 2-rank CUDA regression test that checks async gather equivalence and exercises `DistributedAttention` with replicated QKV. |
| `bcdeca62` | `[test]: add async SP gather benchmark artifacts` | Added the local microbenchmark, the Modal GPU launcher used for L40S runs, and this benchmark/result record. |

## Motivation

`DistributedAttention` supports a replicated-token path, typically for text
tokens that are supplied to every sequence-parallel rank. Before this branch,
the replicated-token output was synchronously gathered across sequence-parallel
ranks immediately after attention. Only after that gather completed did the
main output run its backend postprocessing, sequence padding, and final
sequence-parallel all-to-all.

That ordering forced a serialization point in inference even though the
replicated-token gather is independent from the main-output all-to-all. The
branch moves the no-grad replicated-token gather onto an async path so NCCL can
overlap it with the main-output redistribution work. The autograd path remains
synchronous because the new async collective is intentionally non-autograd.

## Implementation Details

The branch adds a small waitable-handle abstraction in
`fastvideo/distributed/device_communicators/base_device_communicator.py`:

- `AsyncCollectiveTensor` is the interface returned by async collectives.
- `CompletedCollectiveTensor` preserves single-rank behavior without launching a collective.
- `AsyncAllGatherTensor` owns the raw gathered buffer, the original input shape,
  the gather dimension, the process group world size, the `Work` handle, and
  the CUDA gather stream.

`DeviceCommunicatorBase.all_gather_async` starts a non-autograd
`dist.all_gather_into_tensor(..., async_op=True)` operation. For CUDA tensors it
uses a cached communicator-side stream, waits for the current stream before
launch, records input/output tensors on the gather stream, and makes the
current stream wait on the gather stream when `wait()` is called. The final
tensor shape is materialized lazily in `AsyncAllGatherTensor.wait()` with the
same rank/dimension layout as the existing synchronous all-gather path.

`GroupCoordinator.all_gather_async` exposes the communicator method and returns
`CompletedCollectiveTensor` when `world_size == 1`. The public helper
`sequence_model_parallel_all_gather_async` was added in
`fastvideo/distributed/communication_op.py`.

`DistributedAttention.forward` now splits replicated output into a local
contiguous tensor and chooses between:

- synchronous `sequence_model_parallel_all_gather` when gradients are enabled
  and the local replicated output requires grad;
- asynchronous `sequence_model_parallel_all_gather_async` otherwise.

For the async case, `DistributedAttention` starts the replicated-output gather,
then postprocesses and pads the main output, runs the main-output
`sequence_model_parallel_all_to_all_4D`, and only then waits for the replicated
gather result.

## Correctness Coverage

The regression test is
`fastvideo/tests/distributed/test_async_sp_gather.py`.

It is a CUDA test guarded by skip conditions for machines without at least two
GPUs. The pytest entrypoint launches a 2-rank `torchrun` worker with
`FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`.

The worker validates two things:

1. `sequence_model_parallel_all_gather_async(...).wait()` returns the same
   tensor as the synchronous `sequence_model_parallel_all_gather` on a
   rank-distinguishable test tensor gathered along the head dimension.
2. `DistributedAttention` runs in no-grad mode with sequence-parallel inputs
   and replicated QKV, producing finite main and replicated outputs with the
   expected shapes.

Expected saved payload from rank 0:

| Field | Expected value |
|---|---|
| `async_gather_shape` | `(1, 3, 4, 4)` |
| `attention_output_shape` | `(1, 4, 4, 32)` |
| `replicated_output_shape` | `(1, 3, 4, 32)` |

## Benchmark Method

The benchmark script is `tests/local_tests/benchmark_async_sp_gather.py`. It is
a communication microbenchmark, not a full-model inference throughput test.

It compares two communication schedules:

- Sync baseline: `all_gather -> all_to_all`
- Async path: `start all_gather_async -> all_to_all -> wait`

All runs used BF16 tensors, `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`, one node,
and five repeats per case. Each reported result is the median milliseconds per
iteration across the five repeats.

| Case | Full sequence | Replicated sequence | Heads | Head dim | Timed iterations | Warmup |
|---|---:|---:|---:|---:|---:|---:|
| `mid_seq_text` | 8192 | 256 | 16 | 128 | 100 | 20 |
| `long_seq_text` | 32768 | 256 | 16 | 128 | 40 | 10 |
| `replicated_heavy` | 8192 | 2048 | 16 | 128 | 60 | 10 |

The branch also adds `fastvideo/tests/modal/launch_l40s_job.py`, a general
Modal GPU launcher used to run these benchmarks on L40S hardware. The launcher
clones a requested repo/commit into Modal, optionally applies a local patch,
optionally installs FastVideo extras or builds kernels, and runs the requested
shell command on L40S or H100 GPU configurations.

## Benchmark Commands

The benchmarked implementation was commit `23e1da29`. The benchmark script was
later committed in `bcdeca62`; during measurement it was applied as local
benchmark tooling around the implementation commit.

Command shape used for each Modal run:

```bash
python -m modal run fastvideo/tests/modal/launch_l40s_job.py \
  --num-gpus <N> \
  --install-extra none \
  --command "python tests/local_tests/benchmark_async_sp_gather.py --world-size <N> --repeats 5"
```

Modal app IDs recorded for the completed L40S runs:

| GPUs | Modal app |
|---:|---|
| 2x L40S | `ap-BXgEfLSYSpkI4KVi7S3QMV` |
| 4x L40S | `ap-SX64GlyQKT7Z99yUuBl8zJ` |
| 8x L40S | `ap-M6271nLet1qak8oql2Yykv` |

## Benchmark Results

| GPUs | Case | Sync | Async | Delta | Change | Speedup |
|---:|---|---:|---:|---:|---:|---:|
| 2 | `mid_seq_text` | 6.9801 ms | 6.9889 ms | +0.0088 ms | 0.13% slower | 0.9987x |
| 2 | `long_seq_text` | 27.3636 ms | 27.3349 ms | -0.0287 ms | 0.10% faster | 1.0010x |
| 2 | `replicated_heavy` | 9.8851 ms | 9.8022 ms | -0.0829 ms | 0.84% faster | 1.0085x |
| 4 | `mid_seq_text` | 3.5498 ms | 3.5517 ms | +0.0019 ms | 0.05% slower | 0.9995x |
| 4 | `long_seq_text` | 15.0555 ms | 15.0867 ms | +0.0312 ms | 0.21% slower | 0.9979x |
| 4 | `replicated_heavy` | 5.8193 ms | 5.8036 ms | -0.0157 ms | 0.27% faster | 1.0027x |
| 8 | `mid_seq_text` | 3.1690 ms | 3.1747 ms | +0.0057 ms | 0.18% slower | 0.9982x |
| 8 | `long_seq_text` | 11.6985 ms | 11.6901 ms | -0.0084 ms | 0.07% faster | 1.0007x |
| 8 | `replicated_heavy` | 5.8924 ms | 5.8854 ms | -0.0070 ms | 0.12% faster | 1.0012x |

## Validation Summary

| Validation | Command or artifact | Result |
|---|---|---|
| Async gather equality and `DistributedAttention` no-grad smoke | `pytest fastvideo/tests/distributed/test_async_sp_gather.py` on a 2-GPU CUDA host | Passed as the regression gate for commit `1c34762a`; test asserts async gather equals sync gather and output shapes match the expected payload above. |
| 2x L40S microbenchmark | Modal app `ap-BXgEfLSYSpkI4KVi7S3QMV` | Completed. Async path ranged from 0.13% slower to 0.84% faster across the three cases. |
| 4x L40S microbenchmark | Modal app `ap-SX64GlyQKT7Z99yUuBl8zJ` | Completed. Async path ranged from 0.21% slower to 0.27% faster across the three cases. |
| 8x L40S microbenchmark | Modal app `ap-M6271nLet1qak8oql2Yykv` | Completed. Async path ranged from 0.18% slower to 0.12% faster across the three cases. |
| Documentation-session local pytest attempt | `pytest fastvideo/tests/distributed/test_async_sp_gather.py -q` on `/tmp/fastvideo-asynchronous-gather-docs` | Did not reach the CUDA skip/test body on this host. Importing `fastvideo/tests/conftest.py` loaded the installed `fastvideo_kernel` Triton path and failed with `RuntimeError: 0 active drivers ([]). There should only be one.` |
| Documentation-session formatting check | `git diff --check` | Passed. |
| Documentation-session pre-commit check | `pre-commit run --files tests/local_tests/benchmark_async_sp_gather_summary.md` | Passed. Project hooks reported no applicable files for the Markdown path except the filename-spacing check, which passed. |

## Interpretation

The measurements are performance-neutral on the tested L40S communication
shapes. The branch removes a forced serialization point and gives NCCL an
opportunity to overlap replicated-output gathering with main-output
redistribution, but the measured deltas stay within benchmark noise except for
small wins on replicated-heavy shapes.

This should be treated as a structural hot-path improvement with targeted
correctness coverage, not as proof of an end-to-end model throughput win. No
full-model inference benchmark was run as part of this branch.
