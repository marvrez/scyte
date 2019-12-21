#include "ops/conv2d.h"

#include "op.h"
#include "blas.h"
#include "logger.h"

#include <assert.h>

int scyte_conv2d_sync_dims(scyte_node* node)
{
    scyte_node* x = node->children[0], *w = node->children[1];
    if(x->num_dims != 4 || w->num_dims != 4) {
        LOG_ERROR("input and weights must have dim 4");
        return 0;
    }
    if(x->shape[1] != w->shape[1]) {
        LOG_ERROR("input channels of filter and input must be the same");
        return 0;
    }
    int* conv_params = (int*)node->params;
    int size = conv_params[0], stride = conv_params[1], padding = conv_params[2];
    int in_h = x->shape[2], in_w = x->shape[3];

    node->num_dims = 4;
    node->shape[0] = w->shape[0]; // num filters
    node->shape[1] = w->shape[1]; // channels
    node->shape[2] = (in_h + 2*padding - size) / stride + 1; // height
    node->shape[3] = (in_w + 2*padding - size) / stride + 1; // width

    node->tmp = calloc(node->shape[1]*node->shape[2]*node->shape[3]*size*size,sizeof(float));
    return 1;
}

static inline void set_conv_params(scyte_node* node, int stride, int padding)
{
    scyte_node* w = node->children[1];
    assert(w->shape[2] == w->shape[3]);
    int size = w->shape[2];
    int* conv_params = (int*)calloc(3, sizeof(int));
    conv_params[0] = size, conv_params[1] = stride, conv_params[2] = padding;
    node->params = conv_params;
    node->params_size = 3*sizeof(int);
}

scyte_node* scyte_conv2d(scyte_node* x, scyte_node* w, int stride, int padding)
{
    scyte_node* node = make_op2_node(CONV2D, x, w);
    // set-up output shape and save parameters
    set_conv_params(node, stride, padding);
    if(!scyte_conv2d_sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

static inline float im2col_get_pixel(float* im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad, col -= pad;
    if(row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

// from https://github.com/pjreddie/darknet/blob/master/src/im2col.c
void im2col(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col)
{
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    int channels_col = channels*ksize*ksize;
    for(int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize, h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for(int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h*stride, im_col = w_offset + w*stride;
                int col_index = (c*height_col + h)*width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
}


void scyte_conv2d_forward(scyte_node* node)
{
    scyte_node* x = node->children[0], *w = node->children[1];
    int batch_size = x->shape[0], in_c = x->shape[1], in_h = x->shape[2], in_w = x->shape[3];
    int num_filters = node->shape[0], out_h = node->shape[2], out_w = node->shape[3];
    int* conv_params = (int*)node->params;
    int size = conv_params[0], stride = conv_params[1], pad = conv_params[2];

    int m = num_filters, k = size*size*in_c, n = out_w*out_h;
    for(int i = 0; i < batch_size; ++i) {
        float* a = w->vals, *b = node->tmp, *c = node->vals + i*n*m;
        float* im = x->vals + i*in_c*in_h*in_w;
        if(size == 1) b = im;
        else im2col(im, in_c, in_h, in_w, size, stride, pad, b);
        gemm_cpu(0, 0, m, n, k, 1.f, a, b, 1.f, c);
    }
}

void scyte_conv2d_backward(scyte_node* node)
{
    return;
}
