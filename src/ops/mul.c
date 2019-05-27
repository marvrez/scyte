#include "ops/mul.h"

#include "blas.h"
#include "op.h"

#include <assert.h>
#include <stdio.h>

static inline int sync_dims(scyte_node* node)
{
    int n0 = scyte_num_elements(node->children[0]);
    int n1 = scyte_num_elements(node->children[1]);
    if(n0 % n1 != 0) {
        fprintf(stderr, "[scyte_mul] dimensions (%d %% %d != 0) were not properly synced, returning NULL\n", n0, n1);
        return 0;
    }
    scyte_copy_dim(node->children[0], node);
    return 1;
}

scyte_node* scyte_mul(scyte_node* x, scyte_node* y)
{
    scyte_node* node = make_op2_node(MULTIPLY, x, y);
    node->forward = scyte_mul_forward, node->backward = scyte_mul_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_mul_forward(scyte_node* node)
{
    scyte_node* operands[2] = { node->children[0], node->children[1] };
    int n0 = scyte_num_elements(operands[0]), n1 = scyte_num_elements(operands[1]);
    assert(n0 >= n1);
    if(n1 == 1) scale_cpu(n0, operands[1]->vals[0], operands[0]->vals, node->vals);
    else {
        for(int i = 0; i < n0; i += n1) {
            mul_cpu(n1, operands[0]->vals + i, operands[1]->vals, node->vals + i);
        }
    }
}

void scyte_mul_backward(scyte_node* node)
{
    scyte_node* operands[2] = { node->children[0], node->children[1] };
    int n0 = scyte_num_elements(operands[0]), n1 = scyte_num_elements(operands[1]);
    if(scyte_has_gradient(operands[0]) && operands[1]->vals != NULL) {
        for(int i = 0; i < n0; i += n1) {
            mul_sum_cpu(n1, node->delta + i, operands[1]->vals, operands[0]->delta + i);
        }
    }
    if(scyte_has_gradient(operands[1]) && operands[0]->vals != NULL) {
        for(int i = 0; i < n0; i += n1) {
            mul_sum_cpu(n1, node->delta + i, operands[0]->vals + i, operands[1]->delta);
        }
    }
}
