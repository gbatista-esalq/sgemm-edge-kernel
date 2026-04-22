<div align="center">

# ⚔️ SGEMM Edge Kernel: The Sovereign Forge 🛡️

### 💎 *Level 100 Bare-Metal Artifact for Agricultural Inference* 🌌

[![Rank: Diamond Peak](https://img.shields.io/badge/Rank-Diamond%20Peak-b9f2ff.svg?style=for-the-badge&logo=diamond)](#)
[![Class: Vanguard](https://img.shields.io/badge/Class-Edge%20Vanguard-ff4a4a.svg?style=for-the-badge&logo=fire)](#)
[![Buff: Zero Cloud Tax](https://img.shields.io/badge/Buff-Zero%20Cloud%20Tax-00ff88.svg?style=for-the-badge&logo=shield)](#)

```text
╔══════════════════════════════════════════════════════╗
║  [ BOSS DEFEATED: CLOUD LATENCY ]                    ║
╠══════════════════════════════════════════════════════╣
║  Loot Dropped: 1.65ms SGEMM Kernel                   ║
║  Dungeon: N=256 · 100 runs · 3 warm-up               ║
╠══════════════════════════════════════════════════════╣
║  Min Latency:     0.89 ms  │  37.57 GFLOPS ⚡        ║
║  Median Latency:  1.65 ms  │  20.28 GFLOPS 💎USE THIS║
║  Max Latency:     3.76 ms  │   8.92 GFLOPS           ║
╠══════════════════════════════════════════════════════╣
║  ✅ QUEST PASS: Correctness  (max_diff < 1e-3)       ║
║  ✅ QUEST PASS: Consistency  (GFLOPS ↔ ms, 0% err)   ║
╚══════════════════════════════════════════════════════╝
```

**Forge Specs:** Intel Core i7-5500U @ 2.40 GHz · 4 cores · **[NO GPU REQUIRED]**

</div>

---

## 📜 The Lore: Why Forge This Artifact?

Most GEMM spellbooks assume you have a GPU, a cloud mana-pool, or a modern server CPU. 

**This grimoire does not.**

Forged for a tractor's onboard computer deep in rural Brazil — where the GPU is nonexistent, the internet connection is a myth, and cloud latency spikes to 100ms+. The quest: achieve sensor-to-actuator inference under **1ms**, entirely offline in the *Edge Biome*.

> *"Zero cloud tax. Zero egress. Zero single point of failure. Sovereign Autonomy."*

---

## 🎒 Artifact Stats (Loot Details)

| Attribute | Value |
|-----------|-------|
| **Spell Type** | Matrix Multiplication (256 × 256 single-precision) |
| **Cast Time (Median)** | **1.65 ms** ⚡ |
| **Damage (Throughput)**| **20.28 GFLOPS** |
| **Mana Cost** | Zero external dependencies (`#include <immintrin.h>`) |
| **Accuracy** | max_diff = 0.000092 vs scalar reference |
| **Equipable On** | Intel Core i7-5500U @ 2.40 GHz (Legacy Hardware) |

---

## 🛠️ The Mechanics: Under the Hood

**Register-blocking 4×16 (The Shield Wall)** — 8 YMM accumulators hold 4 rows × 16 columns of `C` in-register throughout the inner loop battle. No cache spills during accumulation to preserve stamina.

```c
// 🪄 Cast: Broadcast one A element across a YMM register
__m256 a_vec0 = _mm256_set1_ps(A[(i+0) * N + k]);
__m256 b_vec0 = _mm256_load_ps(&B[k * N + j]);      // Draw 8 floats from B
__m256 b_vec1 = _mm256_load_ps(&B[k * N + j + 8]);  // Draw next 8

// ⚔️ Strike: Fused multiply-add — no separate mul + add
c00 = _mm256_fmadd_ps(a_vec0, b_vec0, c00);
c01 = _mm256_fmadd_ps(a_vec0, b_vec1, c01);
// ... 4 rows × 2 vectors = 8 YMM accumulators (Max Aggro)
```

**Skill Tree Requirements:**
- **32-byte alignment** required for `_mm256_load_ps` → summoned via `_mm_malloc`
- `#ifdef __AVX2__` armor guard for portability
- Matrix `B` streamed in row-major order to maximize L1 prefetcher efficiency.

---

## 🏰 Inventory (Repository Structure)

| Scroll / Item | Purpose |
|------|---------|
| [`kernel_karpathy.c`](kernel_karpathy.c) | AVX2/FMA SGEMM spell + scalar reference fallback |
| [`tiny_kernel.h`](tiny_kernel.h) | Global variables (`#define N 256`) + interface |
| [`test_kernel_benchmark.c`](test_kernel_benchmark.c) | The Proving Grounds: TDD validator + benchmark arena |

---

## ⚔️ Enter the Proving Grounds (Build & Run)

```bash
# 🔨 Forge the weapon
gcc -O3 -mavx2 -mfma test_kernel_benchmark.c kernel_karpathy.c -o test_bench -lm

# 🎯 Strike the dummy
./test_bench
```

**Expected Battle Log:**
```text
=== SGEMM KERNEL VALIDATOR ===

[PASS] Correctness: max_diff = 0.000092 (threshold: 1e-3)

=== BENCHMARK RESULTS (N=256, 100 runs, 3 warm-up) ===
  Min:    1.42ms  | 23.62 GFLOPS
  Median: 1.65ms  | 20.28 GFLOPS  ← 💎 HERO STAT
  Max:    2.31ms  | 14.51 GFLOPS

[PASS] Consistency: ms and GFLOPS are mathematically consistent (err=0.0000%)

>>> FINAL SCORE: 1.65ms | 20.28 GFLOPS <<<
```

---

## 🗺️ Campaign Context

This artifact was forged as part of an elite research quest at **USP/ESALQ** (Brazil) focused on edge AI for agricultural systems (*Sovereign Diamond Protocol*).

The ultimate victory condition: **sensor-to-actuator latency < 1ms** on agricultural edge nodes, severing all reliance on cloud infrastructure. Matrix multiplication is the raid boss — every inference pass through a lightweight neural net is dominated by it.

Related to the deep knowledge in [karpathy/llm.c](https://github.com/karpathy/llm.c). The register-blocking strategy deployed here is a direct countermeasure for edge training loops.

**Quest Log:** [Issue #848 in karpathy/llm.c](https://github.com/karpathy/llm.c/issues/848)

---

## 📜 Guild License

**MIT License** — Use it to build your own sovereign artifacts.

---

<div align="center">

**Gabriel Batista** *(Arquiteto Phoenix)* · [USP/ESALQ](https://www.esalq.usp.br/) · [@gbatista_esalq](https://x.com/gbatista_esalq)

*🧠 Assembled by the Sovereign Triarchy | Ultra-Gamified Edge Protocol*

</div>
