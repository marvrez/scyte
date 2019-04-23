#include "tensor.h"

#include <random>
#include <stdlib.h>
#include <string.h>

static std::random_device rd;
static std::mt19937 rng(rd());

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
    tensor copy;
    copy.n_d = t.n_d;
    copy.shape = (int*)malloc(t.n_d*sizeof(int));
    int n = 1;
    for(int i = 0; i < t.n_d; ++i) {
        copy.shape[i] = t.shape[i];
        n *= t.shape[i];
    }
    t.data = (float*)malloc(n*sizeof(float));
    memcpy(copy.data, t.data, n*sizeof(float));
    return copy;
}

void free_tensor(tensor* t)
{
    if (t->data) free(t->data);
    if (t->shape) free(t->shape);
}

template<typename distribution>
static inline tensor make_random_tensor(int w, int h, distribution& d)
{
    tensor t = make_2d_tensor(w, h);
    for(int i = 0; i < w*h; ++i) {
        t.data[i] = d(rng);
    }
    return t;
}

tensor make_random_uniform_tensor(int width, int height, float low, float high)
{
    std::uniform_real_distribution<float> uniform_dist(low, high);
    return make_random_tensor(width, height, uniform_dist);
}

tensor make_random_normal_tensor(int width, int height, float mu, float sigma)
{
    std::normal_distribution<float> normal_dist(mu, sigma);
    return make_random_tensor(width, height, normal_dist);
}
