#include "ops/exp.h"

#include "blas.h"
#include "op.h"

static inline int sync_dims(scyte_node* node)
{
    scyte_copy_dim(node->children[0], node);
    return 1;
}

scyte_node* scyte_exp(scyte_node* x)
{
    scyte_node* node = make_op1_node(EXP, x);
    node->forward = scyte_exp_forward, node->backward = scyte_exp_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_exp_forward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    exp_cpu(n, operand->vals, node->vals);
}

void scyte_exp_backward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    if(scyte_has_gradient(operand)) {
        for(int i = 0; i < n; ++i) {
            operand->delta[i] += node->delta[i]*operand->vals[i];
        }
    }
}
