#include "blas.h"

#include <math.h>
#include <string.h>

#ifdef OPENBLAS

#ifdef _cplusplus
extern "C" {
#endif

#include <cblas.h>

#ifdef _cplusplus
}
#endif

void gemm_cpu(int trans_a, int trans_b, int M, int N, int K,
        float alpha, const float* A, const float* B, float beta, float* C)
{
    int lda = trans_a ? M : K;
    int ldb = trans_b ? K : N;
    int ldc = N;
    cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemv_cpu(int trans_a, int M, int N, float alpha, 
        const float* A, const float* x, float beta, float* y)
{
    cblas_sgemv(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans, M, N, alpha, A, N, x, 1, beta, y, 1);
}

void axpy_cpu(int N, float alpha, const float* X, float* Y)
{
    cblas_saxpy(N, alpha, X, 1, Y, 1);
}

void axpby_cpu(int N, float alpha, const float* X, float beta, float* Y)
{
    cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

void scale_cpu(int n, float alpha, const float* x, float* y)
{
    cblas_scopy(n, x, 1, y, 1);
    cblas_sscal(n, alpha, y, 1);
}

#else
static inline void gemm_nn(int M, int N, int K, float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float* C, int ldc)
{
    #pragma omp parallel for
    for(int i = 0; i < M; ++i) {
        for(int k = 0; k < K; ++k) {
            float a_part = alpha*A[i*lda+k];
            for(int j = 0; j < N; ++j) {
                C[i*ldc+j] += a_part*B[k*ldb+j];
            }
        }
    }
}

static inline void gemm_nt(int M, int N, int K, float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float* C, int ldc)
{
    #pragma omp parallel for
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float sum = 0;
            for(int k = 0; k < K; ++k) {
                sum += alpha*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

static inline void gemm_tn(int M, int N, int K, float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float* C, int ldc)
{
    #pragma omp parallel for
    for(int i = 0; i < M; ++i) {
        for(int k = 0; k < K; ++k) {
            float a_part = alpha*A[k*lda+i];
            for(int j = 0; j < N; ++j) {
                C[i*ldc+j] += a_part*B[k*ldb+j];
            }
        }
    }
}

static inline void gemm_tt(int M, int N, int K, float alpha,
            const float* A, int lda,
            const float* B, int ldb,
            float* C, int ldc)
{
    #pragma omp parallel for
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float sum = 0;
            for(int k = 0; k < K; ++k) {
                sum += alpha*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int trans_a, int trans_b, int M, int N, int K,
        float alpha, const float* A, const float* B, float beta, float* C)
{
    int lda = trans_a ? M : K;
    int ldb = trans_b ? K : N;
    int ldc = N;

    #pragma omp parallel for
    for(int i = 0; i < M*N; ++i) {
        C[i] *= beta;
    }

    if(!trans_a && !trans_b) gemm_nn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else if(trans_a && !trans_b) gemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else if(!trans_a && trans_b) gemm_nt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else gemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

static inline void gemv_n(int M, int N, float alpha, 
        const float* A, int lda, const float* x, float* y)
{
    #pragma omp parallel for
    for(int i = 0; i < M; ++i) {
        float sum = 0;
        for(int j = 0; j < N; ++j) {
            sum += alpha*A[j+i*lda]*x[j];
        }
        y[i] += sum;
    }
}

static inline void gemv_t(int M, int N, float alpha, 
        const float* A, int lda, const float* x, float* y)
{
    #pragma omp parallel for
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            y[j] += alpha*A[i+j*lda]*x[j];
        }
    }
}

void gemv_cpu(int trans_a, int M, int N, float alpha, 
        const float* A, const float* x, float beta, float* y)
{
    int lda = trans_a ? N : M;
    #pragma omp parallel for
    for(int i = 0; i < lda; ++i) {
        y[i] *= beta;
    }

    if(trans_a) gemv_t(M, N, alpha, A, lda, x, y);
    else gemv_n(M, N, alpha, A, lda, x, y);
}

void axpy_cpu(int N, float alpha, const float* X, float* Y)
{
    #pragma omp parallel for
    for(int i = 0; i < N; ++i) {
        Y[i] += alpha*X[i];
    }
}

void axpby_cpu(int N, float alpha, const float* X, float beta, float* Y)
{
    #pragma omp parallel for
    for(int i = 0; i < N; ++i) {
        Y[i] = alpha*X[i] + beta*Y[i];
    }
}

void scale_cpu(int n, float alpha, const float* x, float* y)
{
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        y[i] = alpha*x[i];
    }
}
#endif

void add_cpu(int n, const float* x, const float* y, float* z)
{
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

void sub_cpu(int n, const float* x, const float* y, float* z)
{
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        z[i] = x[i] - y[i];
    }
}

void mul_cpu(int n, const float* x, const float* y, float* z)
{
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        z[i] = x[i]*y[i];
    }
}

void div_cpu(int n, const float* x, const float* y, float* z)
{
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        z[i] = x[i]/y[i];
    }
}

void mul_sum_cpu(int n, const float* x, const float* y, float* z)
{
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        z[i] += x[i]*y[i];
    }
}

void pow_cpu(int n, float alpha, const float* x, float* y)
{
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        y[i] = pow(x[i], alpha);
    }
}

void bias_cpu(int n, float alpha, const float* x, float* y)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        y[i] = alpha + x[i];
    }
}

void copy_cpu(int N, const float* X, float* Y)
{
    if (X == Y) return;
    memcpy(Y, X, sizeof(float)*N);
}

void exp_cpu(int n, const float* x, float* y)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        y[i] = expf(x[i]);
    }
}

void abs_cpu(int n, const float* x, float* y)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        y[i] = fabsf(x[i]);
    }
}

void set_cpu(int N, float alpha, float* y)
{
    if(alpha == 0.f) {
        memset(y, 0.f, sizeof(float)*N);
        return;
    }
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        y[i] = alpha;
    }
}
