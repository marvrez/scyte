#include "ops/log.h"

#include "blas.h"
#include "op.h"

#include <math.h>

static inline int sync_dims(scyte_node* node)
{
    scyte_copy_dim(node->children[0], node);
    return 1;
}

scyte_node* scyte_log(scyte_node* x)
{
    scyte_node* node = make_op1_node(LOG, x);
    node->forward = scyte_log_forward, node->backward = scyte_log_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_log_forward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    for(int i = 0; i < n; ++i) {
        node->vals[i] = logf(operand->vals[i]);
    }
}

void scyte_log_backward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    if(scyte_has_gradient(operand)) {
        for(int i = 0; i < n; ++i) {
            operand->delta[i] += node->delta[i] / operand->vals[i];
        }
    }
}
