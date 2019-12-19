#include "ops/add.h"

#include "blas.h"
#include "logger.h"
#include "op.h"

#include <assert.h>
#include <stdio.h>

int scyte_add_sync_dims(scyte_node* node)
{
    int n0 = scyte_num_elements(node->children[0]);
    int n1 = scyte_num_elements(node->children[1]);
    if(n0 % n1 != 0) {
        LOG_ERRORF("dimensions (%d %% %d != 0) were not properly synced, returning NULL\n", n0, n1);
        return 0;
    }
    scyte_copy_shape(node->children[0], node);
    return 1;
}

scyte_node* scyte_add(scyte_node* x, scyte_node* y)
{
    scyte_node* node = make_op2_node(ADD, x, y);
    node->forward = scyte_add_forward, node->backward = scyte_add_backward;
    if(!scyte_add_sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_add_forward(scyte_node* node)
{
    scyte_node* operands[2] = { node->children[0], node->children[1] };
    int n0 = scyte_num_elements(operands[0]), n1 = scyte_num_elements(operands[1]);
    assert(n0 >= n1);
    copy_cpu(n0, operands[0]->vals, node->vals);
    for(int i = 0; i < n0; i += n1) {
        axpy_cpu(n1, 1.0f, operands[1]->vals, node->vals + i);
    }
}

void scyte_add_backward(scyte_node* node)
{
    scyte_node* operands[2] = { node->children[0], node->children[1] };
    int n0 = scyte_num_elements(operands[0]), n1 = scyte_num_elements(operands[1]);
    if(scyte_has_gradient(operands[0])) {
        axpy_cpu(n0, 1.0f, node->delta, operands[0]->delta);
    }
    if(scyte_has_gradient(operands[1])) {
        for(int i = 0; i < n0; i += n1) {
            axpy_cpu(n1, 1.0f, node->delta + i, operands[1]->delta);
        }
    }
}
