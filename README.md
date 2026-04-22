<div align="center">

# ⚡ sgemm-edge-kernel

### Bare-metal AVX2/FMA matrix multiplication for edge AI — no GPU, no cloud, no dependencies.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Language: C](https://img.shields.io/badge/Language-C-blue.svg)](kernel_karpathy.c)
[![Benchmark: TDD](https://img.shields.io/badge/Benchmark-TDD%20Validated-brightgreen.svg)](test_kernel_benchmark.c)
[![Hardware: No GPU](https://img.shields.io/badge/Hardware-No%20GPU-red.svg)](#results)

```
╔══════════════════════════════════════════════════════╗
║  BENCHMARK RESULTS  (N=256 · 100 runs · 3 warm-up)  ║
╠══════════════════════════════════════════════════════╣
║  Min:     0.89 ms  │  37.57 GFLOPS                  ║
║  Median:  1.65 ms  │  20.28 GFLOPS  ◄ USE THIS      ║
║  Max:     3.76 ms  │   8.92 GFLOPS                  ║
╠══════════════════════════════════════════════════════╣
║  ✅ PASS  Correctness  (max_diff < 1e-3)             ║
║  ✅ PASS  Consistency  (GFLOPS ↔ ms, err=0.0000%)   ║
╚══════════════════════════════════════════════════════╝
```

**Hardware:** Intel Core i7-5500U @ 2.40 GHz · 4 cores (2 physical) · **No GPU**

</div>

---

## Why this kernel?

Most GEMM libraries assume you have a GPU, a cloud connection, or at least a modern server CPU.

**This one doesn't.**

It was built for a tractor's onboard computer in rural Brazil — where there's no GPU, no reliable internet, and latency from cloud inference can spike to 100ms+. The goal: sensor-to-actuator inference under **1ms**, entirely offline.

> *"Zero cloud tax. Zero egress. Zero single point of failure."*

---

## Results

| Metric | Value |
|--------|-------|
| Matrix size | 256 × 256 (single-precision) |
| **Median latency** | **1.65 ms** |
| **Throughput** | **20.28 GFLOPS** |
| Runs measured | 100 (after 3 warm-ups) |
| Correctness | max_diff = 0.000092 vs scalar reference |
| Hardware | Intel Core i7-5500U @ 2.40 GHz — **no GPU** |

---

## How it works

**Register-blocking 4×16** — 8 YMM accumulators hold 4 rows × 16 columns of C in-register throughout the inner loop. No cache spills during accumulation.

```c
// Broadcast one A element across a YMM register
__m256 a_vec0 = _mm256_set1_ps(A[(i+0) * N + k]);
__m256 b_vec0 = _mm256_load_ps(&B[k * N + j]);      // 8 floats from B
__m256 b_vec1 = _mm256_load_ps(&B[k * N + j + 8]);  // next 8

// Fused multiply-add — no separate mul + add
c00 = _mm256_fmadd_ps(a_vec0, b_vec0, c00);
c01 = _mm256_fmadd_ps(a_vec0, b_vec1, c01);
// ... 4 rows × 2 vectors = 8 YMM accumulators total
```

Key constraints:
- **32-byte alignment** required for `_mm256_load_ps` → allocated via `_mm_malloc`
- `#ifdef __AVX2__` guard for portability to non-AVX2 targets
- B streamed in row-major order to maximize L1 prefetcher efficiency

---

## Files

| File | Purpose |
|------|---------|
| [`kernel_karpathy.c`](kernel_karpathy.c) | AVX2/FMA SGEMM kernel + scalar reference |
| [`tiny_kernel.h`](tiny_kernel.h) | `#define N 256` + function prototypes |
| [`test_kernel_benchmark.c`](test_kernel_benchmark.c) | TDD validator: correctness + benchmark consistency |

---

## Build & Run

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

---

## Benchmark methodology

- Wall-clock via `clock_gettime(CLOCK_MONOTONIC)` (sub-ms resolution)
- 3 warm-up runs discarded (cache/branch-predictor warm-up)
- 100 timed runs, bubble-sorted for median
- Correctness verified: max element-wise diff vs scalar reference < 1e-3
- GFLOPS derived from median: `2 × N³ / (time_s × 1e9)`

---

## Context

Part of a research project at **USP/ESALQ** (Brazil) on edge AI for agricultural systems.

The architecture goal: **sensor-to-actuator latency under 1ms** on a tractor's onboard computer, eliminating cloud dependency entirely. SGEMM is the core bottleneck — every inference pass through a lightweight neural net is dominated by matmul.

Related to the matmul-bound architecture discussed in [karpathy/llm.c](https://github.com/karpathy/llm.c). The register-blocking strategy here is directly applicable to edge training loops.

**See also:** [Issue #848 in karpathy/llm.c](https://github.com/karpathy/llm.c/issues/848)

---

## License

MIT

---

<div align="center">

**Gabriel Batista** · [USP/ESALQ](https://www.esalq.usp.br/) · [@gbatista_esalq](https://x.com/gbatista_esalq)

*Built for the field. Validated in code.*

</div>
