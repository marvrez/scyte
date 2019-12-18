#include "ops/square.h"

#include "blas.h"
#include "op.h"

static inline int sync_dims(scyte_node* node)
{
    scyte_copy_shape(node->children[0], node);
    return 1;
}

scyte_node* scyte_square(scyte_node* x)
{
    scyte_node* node = make_op1_node(SQUARE, x);
    node->forward = scyte_square_forward, node->backward = scyte_square_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_square_forward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    mul_cpu(n, operand->vals, operand->vals, node->vals);
}

void scyte_square_backward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    if(scyte_has_gradient(operand)) {
        for(int i = 0; i < n; ++i) {
            operand->delta[i] += node->delta[i]*(2*operand->vals[i]);
        }
    }
}
