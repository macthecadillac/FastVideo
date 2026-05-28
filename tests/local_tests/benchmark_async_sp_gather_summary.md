# Async Sequence-Parallel Gather Benchmark

This records the Modal L40S measurements for commit `23e1da29`, which changes
`DistributedAttention` to start the replicated-token sequence-parallel
`all_gather` asynchronously during no-grad inference and wait after the main
output all-to-all.

The benchmark script is `tests/local_tests/benchmark_async_sp_gather.py`. It
isolates the communication sequence:

- sync baseline: `all_gather -> all_to_all`
- async path: `start all_gather_async -> all_to_all -> wait`

These are microbenchmarks, not full-model inference throughput results.

## Modal Runs

| GPUs | Modal app |
|---:|---|
| 2x L40S | `ap-BXgEfLSYSpkI4KVi7S3QMV` |
| 4x L40S | `ap-SX64GlyQKT7Z99yUuBl8zJ` |
| 8x L40S | `ap-M6271nLet1qak8oql2Yykv` |

All runs used five repeats per case.

## Results

| GPUs | Case | Sync | Async | Change |
|---:|---|---:|---:|---:|
| 2 | `mid_seq_text` | 6.9801 ms | 6.9889 ms | 0.13% slower |
| 2 | `long_seq_text` | 27.3636 ms | 27.3349 ms | 0.10% faster |
| 2 | `replicated_heavy` | 9.8851 ms | 9.8022 ms | 0.84% faster |
| 4 | `mid_seq_text` | 3.5498 ms | 3.5517 ms | 0.05% slower |
| 4 | `long_seq_text` | 15.0555 ms | 15.0867 ms | 0.21% slower |
| 4 | `replicated_heavy` | 5.8193 ms | 5.8036 ms | 0.27% faster |
| 8 | `mid_seq_text` | 3.1690 ms | 3.1747 ms | 0.18% slower |
| 8 | `long_seq_text` | 11.6985 ms | 11.6901 ms | 0.07% faster |
| 8 | `replicated_heavy` | 5.8924 ms | 5.8854 ms | 0.12% faster |

## Conclusion

The change is performance-neutral on the tested L40S shapes. It removes a
forced serialization point and gives NCCL a chance to overlap the replicated
gather with main-output redistribution, but the measured delta stays within
noise except for tiny wins on replicated-heavy shapes.

Treat this as a structural improvement with correctness coverage, not as a
proven end-to-end throughput win.
