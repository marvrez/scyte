#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    int n_d;        // number of dimensions
    int* shape;     // shape, of size n_d
    float* data;    // array data, of size prod_i(d[i]) (1 if n_d==0)
} tensor;

inline int tensor_length(tensor t) 
{
    int n = 1;
    for(int i = 0; i < t.n_d; ++i) {
        n *= t.shape[i];
    }
    return n;
}

tensor make_tensor(int n_d, int* shape);
tensor make_2d_tensor(int width, int height);
tensor make_3d_tensor(int width, int height, int channels);
tensor copy_tensor(tensor t);
void free_tensor(tensor* t);

tensor make_random_uniform_tensor(int width, int height, float low, float high);
tensor make_random_normal_tensor(int width, int height, float mu, float sigma);

#endif
