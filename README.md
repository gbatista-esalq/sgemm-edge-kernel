# sgemm-edge-kernel

Dependency-free single-precision GEMM in C using AVX2/FMA intrinsics,
built for edge AI inference on legacy agricultural hardware.

No GPU. No cloud. No external libraries.

## Results

| Metric | Value |
|--------|-------|
| Matrix size | 256 × 256 (single-precision) |
| Median latency | **1.65 ms** |
| Throughput | **20.28 GFLOPS** |
| Runs measured | 100 (after 3 warm-ups) |
| Hardware | Intel Core i7-5500U @ 2.40 GHz — no GPU |

## Optimization strategy

**Register blocking 4×16** — 8 YMM accumulators (c00..c31) saturate the FMA
pipeline; 2 YMM registers stream B, 1 broadcast per A-element.

```c
// Broadcast A, stream B in L1-cache order — saturates YMM registers
__m256 a  = _mm256_set1_ps(A[(i+0) * N + k]);
__m256 b0 = _mm256_load_ps(&B[k * N + j]);
__m256 b1 = _mm256_load_ps(&B[k * N + j + 8]);

c00 = _mm256_fmadd_ps(a, b0, c00);
c01 = _mm256_fmadd_ps(a, b1, c01);
// ... 4 rows × 16 cols = 8 YMM accumulators
```

Key constraints:
- 32-byte alignment required for `_mm256_load_ps` — allocated via `_mm_malloc`
- `#ifdef __AVX2__` guard for portability to non-AVX2 targets
- Inner loop streams B in-order for L1 prefetcher efficiency

## Files

| File | Purpose |
|------|---------|
| `kernel_karpathy.c` | AVX2/FMA SGEMM kernel + scalar reference |
| `tiny_kernel.h` | `#define N 256` + function prototypes |
| `test_kernel_benchmark.c` | TDD validator: correctness + benchmark consistency |

## Build & run

```bash
# Compile
gcc -O3 -mavx2 -mfma test_kernel_benchmark.c kernel_karpathy.c -o test_bench -lm

# Run
./test_bench
```

Expected output:
```
=== SGEMM KERNEL VALIDATOR ===

[PASS] Correctness: max_diff = 0.000092 (threshold: 1e-3)

=== BENCHMARK RESULTS (N=256, 100 runs, 3 warm-up) ===
  Min:    1.42ms  | 23.62 GFLOPS
  Median: 1.65ms  | 20.28 GFLOPS  ← USE THIS IN REPORTS
  Max:    2.31ms  | 14.51 GFLOPS

[PASS] Consistency: ms and GFLOPS are mathematically consistent (err=0.0000%)

>>> NUMBERS FOR REPORTS: 1.65ms | 20.28 GFLOPS <<<
```

## Benchmark methodology

- Wall-clock via `clock_gettime(CLOCK_MONOTONIC)` (sub-ms resolution)
- 3 warm-up runs discarded (cache/branch-predictor warm-up)
- 100 timed runs, bubble-sorted for median
- Correctness verified: max element-wise diff vs scalar reference < 1e-3
- GFLOPS derived from median: `2 × N³ / (time_s × 1e9)`

## Context

Part of a research project at USP/ESALQ (Brazil) on edge AI for agricultural
systems. The goal is sensor-to-actuator latency under 1 ms on a tractor's
onboard computer, eliminating cloud dependency entirely.

SGEMM is the core bottleneck: every inference pass through a lightweight
neural net is dominated by matmul. This kernel targets legacy rural hardware
where no GPU or neural accelerator is available.

Related to the architecture discussed in [llm.c](https://github.com/karpathy/llm.c)
— training loops have the same matmul dominance. The register-blocking strategy
here is directly applicable.

## License

MIT

## Author

Gabriel Batista — USP/ESALQ
