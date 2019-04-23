#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    int n_d;        // number of dimensions
    int* shape;     // shape, of size n_d
    float* data;    // array data, of size prod_i(d[i]) (1 if n_d==0)
} tensor;

tensor make_tensor(int n_d, int* shape);
tensor make_2d_tensor(int width, int height);
tensor make_3d_tensor(int width, int height, int channels);
tensor copy_tensor(tensor t);
void free_tensor(tensor* t);

tensor make_random_uniform_tensor(int width, int height, float low=0.f, float high=1.f);
tensor make_random_normal_tensor(int width, int height, float mu=0.f, float sigma=1.f);

#endif
