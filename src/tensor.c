#include "tensor.h"

#include "utils.h"

#include <stdlib.h>
#include <string.h>

tensor make_tensor(int n_d, int* shape)
{
    tensor t;
    t.n_d = n_d;
    t.shape = (int*)malloc(n_d*sizeof(int));
    int n = 1;
    for(int i = 0; i < n_d; ++i) {
        t.shape[i] = shape[i];
        n *= shape[i];
    }
    t.data = (float*)calloc(n, sizeof(float*));
    return t;
}

tensor make_2d_tensor(int width, int height)
{
    int shape[2] = {width, height};
    return make_tensor(2, shape);
}

tensor make_3d_tensor(int width, int height, int channels)
{
    int shape[3] = {width, height, channels};
    return make_tensor(3, shape);
}

tensor copy_tensor(tensor t)
{
    tensor copy = make_tensor(t.n_d, t.shape);
    int n = tensor_length(t);
    memcpy(copy.data, t.data, n*sizeof(float));
    return copy;
}

void free_tensor(tensor* t)
{
    if (t->data) free(t->data);
    if (t->shape) free(t->shape);
}

static inline tensor make_random_tensor(int w, int h, float a, float b, float(*f)(float, float))
{
    tensor t = make_2d_tensor(w, h);
    for(int i = 0; i < w*h; ++i) {
        t.data[i] = f(a,b);
    }
    return t;
}

tensor make_random_uniform_tensor(int width, int height, float low, float high)
{
    return make_random_tensor(width, height, low, high, random_uniform);
}

tensor make_random_normal_tensor(int width, int height, float mu, float sigma)
{
    return make_random_tensor(width, height, mu, sigma, random_normal);
}
