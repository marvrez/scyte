#include "ops/normalize.h"

#include "blas.h"
#include "op.h"

#include <math.h>
#include <stdlib.h>
#include <assert.h>

#define EPS 1e-6

// TODO: generalize so that same operation can be used for batch norm too
//       e.g. make `axis` a parameter

int scyte_normalize_sync_dims(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    assert(operand->num_dims > 0);
    scyte_copy_shape(operand, node);
    int batch_size = scyte_num_elements(operand) / operand->shape[operand->num_dims - 1];
    node->tmp = realloc(node->tmp, batch_size*sizeof(float));
    return 1;
}

scyte_node* scyte_normalize(scyte_node* x)
{
    scyte_node* node = make_op1_node(NORMALIZE, x);
    node->forward = scyte_normalize_forward, node->backward = scyte_normalize_backward;
    if(!scyte_normalize_sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_normalize_forward(scyte_node* node)
{
    scyte_node* in = node->children[0];
    assert(node->tmp);
    int i, j;
    int dim = in->shape[in->num_dims - 1];
    int batch_size = scyte_num_elements(in) / dim;

    float* std_inv = (float*)node->tmp;
    for(i = 0; i < batch_size; ++i) {
        float* in_vals  = &in->vals[i*dim];
        float* out_vals = &node->vals[i*dim];
        float mu = 0.f, sse = 0.f, sigma_inv;

        for(j = 0; j < dim; ++j) mu += in_vals[j];
        mu /= dim;
        for(j = 0; j < dim; ++j) {
            out_vals[j] = in_vals[j] - mu;
            sse += out_vals[j]*out_vals[j];
        }
        sigma_inv = sse == 0.f ? 1.f : 1.f / sqrtf(sse / (float)dim + EPS);
        for(j = 0; j < dim; ++j) out_vals[j] *= sigma_inv;

        std_inv[i] = sigma_inv;
    }
}

void scyte_normalize_backward(scyte_node* node)
{
    scyte_node* in = node->children[0];
    assert(node->tmp);
    if(!scyte_has_gradient(in)) return;
    int i, j;
    int dim = in->shape[in->num_dims - 1];
    int batch_size = scyte_num_elements(in) / dim;

    float* std_inv = (float*)node->tmp;
    for(i = 0; i < batch_size; ++i) {
        float* in_delta   = &in->delta[i*dim];
        float* out_delta  = &node->delta[i*dim];
        float* out_vals   = &node->vals[i*dim];
        float mu_delta = 0.f, s = 0.f, sigma_inv = std_inv[i];
        
        for(j = 0; j < dim; ++j) {
            mu_delta += out_delta[j];
            s += out_vals[j]*out_delta[j];
        }
        mu_delta /= dim, s /= dim;
        for(j = 0; j < dim; ++j) {
            in_delta[j] += sigma_inv*(out_delta[j] - mu_delta - s*out_vals[j]);
        }
    }
}
