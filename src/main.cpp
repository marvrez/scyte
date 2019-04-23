#include "blas.h"
#include "tensor.h"
#include "utils.h"
#include <cstdio>
#include <math.h>

int main(int argc, char** argv)
{
    int M = 300, N = 300, K = 300;
    tensor a  = make_random_uniform_tensor(M, K);
    tensor b = make_random_uniform_tensor(K, N);
    tensor c = make_2d_tensor(M, N);

    double t1 = time_now();
    gemm_cpu(false, false, M, N, K, 1, a.data, b.data, 1, c.data);
    double t2 = time_now();
    printf("time taken: %.6lf miliseconds\n", (t2-t1)*100);
    return 0;
}
