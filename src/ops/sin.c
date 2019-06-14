#include "ops/sin.h"

#include "blas.h"
#include "op.h"

#include <math.h>

static inline int sync_dims(scyte_node* node)
{
    scyte_copy_shape(node->children[0], node);
    return 1;
}

scyte_node* scyte_sin(scyte_node* x)
{
    scyte_node* node = make_op1_node(SIN, x);
    node->forward = scyte_sin_forward, node->backward = scyte_sin_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_sin_forward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        node->vals[i] = sinf(operand->vals[i]);
    }
}

void scyte_sin_backward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    if(scyte_has_gradient(operand)) {
        #pragma omp parallel for
        for(int i = 0; i < n; ++i) {
            operand->delta[i] += node->delta[i]*cosf(operand->vals[i]);
        }
    }
}
