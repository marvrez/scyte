#include "ops/tanh.h"

#include "blas.h"
#include "op.h"

#include <math.h>

static inline int sync_dims(scyte_node* node)
{
    scyte_copy_dim(node->children[0], node);
    return 1;
}

scyte_node* scyte_tanh(scyte_node* x)
{
    scyte_node* node = make_op1_node(TANH, x);
    node->forward = scyte_tanh_forward, node->backward = scyte_tanh_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_tanh_forward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    float y;
    for(int i = 0; i < n; ++i) {
        y = expf(-2.f*operand->vals[i]);
        node->vals[i] = (1.f - y) / (1.f + y);
    }
}

void scyte_tanh_backward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    if(scyte_has_gradient(operand)) {
        float tanh_val;
        for(int i = 0; i < n; ++i) {
            tanh_val = operand->vals[i];
            operand->delta[i] += node->delta[i]*(1.f - tanh_val*tanh_val);
        }
    }
}
