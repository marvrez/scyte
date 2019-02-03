#ifndef BLAS_H
#define BLAS_H

void gemm_cpu(bool trans_a, bool trans_b, int M, int N, int K,
        float alpha, const float* A, const float* B, float beta, float* C);

void gemv_cpu(bool trans_a, int M, int N, float alpha, 
        const float* A, const float* x, const float beta, float* y);

void axpy_cpu(int N, float alpha, const float* X, float* Y);
void axpby_cpu(int N, float alpha, const float* X, float beta, float* Y);

void copy_cpu(int N, const float* X, float* Y);
void set_cpu(int N, float alpha, float* y);

void add_cpu(int n, const float* x, const float* y, float* z);
void sub_cpu(int n, const float* x, const float* y, float* z);
void mul_cpu(int n, const float* x, const float* y, float* z);
void div_cpu(int n, const float* x, const float* y, float* z);

void pow_cpu(int n, float alpha, const float* x, float* y);
void scale_cpu(int n, float alpha, const float* x, float* y);
void bias_cpu(int n, float alpha, const float* x, float* y);
void exp_cpu(int n, const float* x, float* y);
void abs_cpu(int n, const float* x, float* y);

#endif
