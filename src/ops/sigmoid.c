#include "ops/sigmoid.h"

#include "blas.h"
#include "op.h"

#include <math.h>

static inline int sync_dims(scyte_node* node)
{
    scyte_copy_shape(node->children[0], node);
    return 1;
}

scyte_node* scyte_sigmoid(scyte_node* x)
{
    scyte_node* node = make_op1_node(SIGMOID, x);
    node->forward = scyte_sigmoid_forward, node->backward = scyte_sigmoid_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    fprintf(stderr, "sigmoid                          %4d\n", scyte_num_elements(x));
    return node;
}

#define sigmoid(x) 0.5f*tanhf(0.5f*x) + 0.5f
void scyte_sigmoid_forward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        node->vals[i] = sigmoid(operand->vals[i]);
    }
}
#undef sigmoid

void scyte_sigmoid_backward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    if(scyte_has_gradient(operand)) {
        float sig;
        #pragma omp parallel for
        for(int i = 0; i < n; ++i) {
            sig = operand->vals[i];
            operand->delta[i] += node->delta[i]*sig*(1.f - sig);
        }
    }
}
