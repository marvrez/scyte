#include "ops/softmax.h"

#include "blas.h"
#include "op.h"

#include <math.h>
#include <float.h>

static inline int sync_dims(scyte_node* node)
{
    scyte_copy_dim(node->children[0], node);
    return 1;
}

scyte_node* scyte_softmax(scyte_node* x)
{
    scyte_node* node = make_op1_node(SOFTMAX, x);
    node->forward = scyte_softmax_forward, node->backward = scyte_softmax_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_softmax_forward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int dim = operand->shape[operand->num_dims - 1];
    int batch_size = scyte_num_elements(operand) / dim;
    for(int i = 0; i < batch_size; ++i) {
        float max = -FLT_MAX, scale = 0.f;
        float* out_vals  = &node->vals[i*dim];
        float* in_vals   = &operand->vals[i*dim];
        for(int j = 0; j < dim; ++j) max = max > in_vals[j] ? max : in_vals[j];
        for(int j = 0; j < dim; ++j) {
            out_vals[j] = expf(in_vals[j] - max);
            scale += out_vals[j];
        }
        scale = 1.f / scale;
        for(int j = 0; j < dim; ++j) out_vals[j] *= scale;
    }
}

//  We assume that probs is of shape [batch_size, dim]
//  The formula for dsoftmax / dx = (diag(softmax) - softmax * softmax').
//  This matrix is diagonal minus a rank one matrix, so it is easy to implement
//  as follows:
//  grad_x = grad_softmax*softmax - sum(grad_softmax * softmax)*softmax
void scyte_softmax_backward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int dim = operand->shape[operand->num_dims - 1];
    int batch_size = scyte_num_elements(operand) / dim;
    if(scyte_has_gradient(operand)) {
        for(int i = 0; i < batch_size; ++i) {
            float* operand_delta = &operand->delta[i*dim];

            float* node_delta = &node->delta[i*dim];
            float* softmax = &node->vals[i*dim];

            float sum = 0.f;
            for(int j = 0; j < dim; ++j) sum += node_delta[j]*softmax[j];
            for(int j = 0; j < dim; ++j) operand_delta[j] += softmax[j]*(node_delta[j] - sum);
        }
    }
}
