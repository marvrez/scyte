#include "ops/maxpool2d.h"

#include "op.h"
#include "logger.h"

#include <stdlib.h>
#include <float.h>

int scyte_maxpool2d_sync_dims(scyte_node* node)
{
    scyte_node* x = node->children[0];
    int n = x->shape[0], c = x->shape[1], h = x->shape[2], w = x->shape[3];
    int* pool_params = (int*)node->params;
    int size = pool_params[0], stride = pool_params[1], padding = pool_params[2];
    node->num_dims = 4;
    node->shape[0] = n;
    node->shape[1] = c;
    node->shape[2] = (h + padding - size)/stride + 1;
    node->shape[3] = (w + padding - size)/stride + 1;

    // tmp will store the max indexes, for use in backward propagation
    node->tmp = realloc(node->tmp, scyte_num_elements(node)*sizeof(int));

    return 1;
}

static inline void set_pool_params(scyte_node* node, int size, int stride, int padding)
{
    int* pool_params = (int*)calloc(3, sizeof(int));
    pool_params[0] = size, pool_params[1] = stride, pool_params[2] = padding;
    node->params = pool_params;
    node->params_size = 3*sizeof(int);
}

scyte_node* scyte_maxpool2d(scyte_node* x, int size, int stride, int padding)
{
    if(x->num_dims != 4) {
        LOG_ERRORF("x has %d dimension(s), it must have shape NCHW", x->num_dims);
        return NULL;
    }
    scyte_node* node = make_op1_node(MAXPOOL2D, x);
    set_pool_params(node, size, stride, padding);
    // set-up output shape and save parameters
    if(!scyte_maxpool2d_sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_maxpool2d_forward(scyte_node* node)
{
    scyte_node* x = node->children[0];
    int in_h = node->shape[2], in_w = node->shape[3];

    int batch = node->shape[0], c = node->shape[1], h = node->shape[2], w = node->shape[3];
    int* indexes = (int*)node->tmp;
    int* pool_params = (int*)node->params;
    int size = pool_params[0], stride = pool_params[1], padding = pool_params[2];
    int w_offset = -padding / 2.f, h_offset = -padding / 2.f;

    for(int b = 0; b < batch; ++b) {
        for(int k = 0; k < c; ++k) {
            for(int i = 0; i < h; ++i) {
                for(int j = 0; j < w; ++j) {
                    int out_idx = j + w*(i + h*(k + c*b)), max_idx = -1;
                    float max_val = -FLT_MAX;
                    for(int n = 0; n < size; ++n) {
                        for(int m = 0; m < size; ++m) {
                            int cur_y = h_offset + i*stride + n, cur_x = w_offset + j*stride + m;
                            int idx = cur_x + in_w*(cur_y + in_h*(k + b*c));
                            int is_valid = (cur_y >= 0 && cur_y < in_h && cur_x >= 0 && cur_x < in_w);
                            float val = is_valid ? x->vals[idx] : -FLT_MAX;
                            if(val > max_val) max_idx = idx, max_val = val;
                        }
                    }
                    node->vals[out_idx] = max_val;
                    indexes[out_idx] = max_idx;
                }
            }
        }
    }
}

void scyte_maxpool2d_backward(scyte_node* node)
{
    scyte_node* x = node->children[0];
    int n = scyte_num_elements(node), *indexes = (int*)node->tmp;
    for(int i = 0; i < n; ++i) {
        int idx = indexes[i];
        x->delta[idx] += node->delta[i];
    }
}
