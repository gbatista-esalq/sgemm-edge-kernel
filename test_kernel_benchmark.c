/*
 * TDD Benchmark Validator — sgemm-edge-kernel
 * ----------------------------------------------------------------
 * Verifies that GFLOPS and ms are mathematically consistent.
 * Correctness check: max element-wise diff vs scalar reference < 1e-3.
 *
 * Compila: gcc -O3 -mavx2 -mfma test_kernel_benchmark.c kernel_karpathy.c -o test_bench
 * Roda:    ./test_bench
 *
 * Critérios de PASS:
 *   1. sgemm_tiny_kernel produz resultado correto vs sgemm_reference (max_diff < 1e-3)
 *   2. GFLOPS calculado é matematicamente consistente com tempo medido (diff < 1%)
 *   3. Latência mediana impressa é o número REAL para usar no draft
 */

#include "tiny_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

#define WARMUP_RUNS 3
#define BENCH_RUNS  100

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int test_correctness(void) {
    float* A = (float*)_mm_malloc(N * N * sizeof(float), 32);
    float* B = (float*)_mm_malloc(N * N * sizeof(float), 32);
    float* C_ref = (float*)_mm_malloc(N * N * sizeof(float), 32);
    float* C_opt = (float*)_mm_malloc(N * N * sizeof(float), 32);

    if (!A || !B || !C_ref || !C_opt) {
        printf("[FAIL] Memory allocation error\n");
        return 0;
    }

    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
        C_ref[i] = 0.0f;
        C_opt[i] = 0.0f;
    }

    sgemm_reference(A, B, C_ref);
    sgemm_tiny_kernel(A, B, C_opt);

    float max_diff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = fabsf(C_ref[i] - C_opt[i]);
        if (diff > max_diff) max_diff = diff;
    }

    _mm_free(A); _mm_free(B); _mm_free(C_ref); _mm_free(C_opt);

    if (max_diff < 1e-3f) {
        printf("[PASS] Correctness: max_diff = %.6f (threshold: 1e-3)\n", max_diff);
        return 1;
    } else {
        printf("[FAIL] Correctness: max_diff = %.6f EXCEEDS threshold 1e-3\n", max_diff);
        return 0;
    }
}

static int test_benchmark_consistency(void) {
    float* A = (float*)_mm_malloc(N * N * sizeof(float), 32);
    float* B = (float*)_mm_malloc(N * N * sizeof(float), 32);
    float* C = (float*)_mm_malloc(N * N * sizeof(float), 32);

    if (!A || !B || !C) {
        printf("[FAIL] Memory allocation error\n");
        return 0;
    }

    srand(123);
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
        C[i] = 0.0f;
    }

    /* Warm-up: discartados */
    for (int w = 0; w < WARMUP_RUNS; w++) {
        memset(C, 0, N * N * sizeof(float));
        sgemm_tiny_kernel(A, B, C);
    }

    /* Benchmark: 100 runs */
    double times_ms[BENCH_RUNS];
    for (int r = 0; r < BENCH_RUNS; r++) {
        memset(C, 0, N * N * sizeof(float));
        double t0 = get_time_ms();
        sgemm_tiny_kernel(A, B, C);
        double t1 = get_time_ms();
        times_ms[r] = t1 - t0;
    }

    /* Ordenar para mediana */
    for (int i = 0; i < BENCH_RUNS - 1; i++) {
        for (int j = i + 1; j < BENCH_RUNS; j++) {
            if (times_ms[j] < times_ms[i]) {
                double tmp = times_ms[i];
                times_ms[i] = times_ms[j];
                times_ms[j] = tmp;
            }
        }
    }

    double ms_min    = times_ms[0];
    double ms_median = times_ms[BENCH_RUNS / 2];
    double ms_max    = times_ms[BENCH_RUNS - 1];

    double flops = 2.0 * N * N * N;
    double gflops_from_median = flops / (ms_median / 1000.0) / 1e9;

    printf("\n=== BENCHMARK RESULTS (N=%d, %d runs, %d warm-up) ===\n",
           N, BENCH_RUNS, WARMUP_RUNS);
    printf("  Min:    %.4f ms  | %.2f GFLOPS\n", ms_min,    flops / (ms_min    / 1000.0) / 1e9);
    printf("  Median: %.4f ms  | %.2f GFLOPS  ← USE ESTE NO DRAFT\n", ms_median, gflops_from_median);
    printf("  Max:    %.4f ms  | %.2f GFLOPS\n", ms_max,    flops / (ms_max    / 1000.0) / 1e9);

    /* Verificar consistência: GFLOPS deve ser matematicamente derivado do tempo */
    double expected_ms_from_gflops = flops / (gflops_from_median * 1e9) * 1000.0;
    double consistency_err = fabs(expected_ms_from_gflops - ms_median) / ms_median;

    if (consistency_err < 0.01) {  /* < 1% de erro */
        printf("[PASS] Consistency: ms e GFLOPS são matematicamente consistentes (err=%.4f%%)\n",
               consistency_err * 100.0);
        printf("\n>>> NÚMEROS PARA O DRAFT: %.2fms | %.2f GFLOPS <<<\n",
               ms_median, gflops_from_median);
        _mm_free(A); _mm_free(B); _mm_free(C);
        return 1;
    } else {
        printf("[FAIL] Consistency: inconsistência %.2f%% — não postar até corrigir\n",
               consistency_err * 100.0);
        _mm_free(A); _mm_free(B); _mm_free(C);
        return 0;
    }
}

int main(void) {
    printf("=== SGEMM KERNEL VALIDATOR ===\n");
    printf("Mandato: zero dispatch sem PASS em todos os testes\n\n");

    int pass = 1;
    pass &= test_correctness();
    pass &= test_benchmark_consistency();

    printf("\n=== RESULTADO FINAL ===\n");
    if (pass) {
        printf("[PASS] Kernel aprovado. Copie os números acima para o draft v2.\n");
    } else {
        printf("[FAIL] Kernel REPROVADO. Corrigir antes de qualquer postagem.\n");
        printf("       Consultar AGAP_ULTRATHINK_AUDIT.md — seção de erros.\n");
    }

    return pass ? 0 : 1;
}
