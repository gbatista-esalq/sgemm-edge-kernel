#ifndef TINY_KERNEL_H
#define TINY_KERNEL_H

#define N 256
void sgemm_reference(const float* A, const float* B, float* C);
void sgemm_tiny_kernel(const float* A, const float* B, float* C);

#endif
