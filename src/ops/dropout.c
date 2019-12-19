#include "ops/dropout.h"

#include "op.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>

int scyte_dropout_sync_dims(scyte_node* node)
{
    scyte_copy_shape(node->children[0], node);
    // allocate space to store which elements were kept
    int n = scyte_num_elements(node->children[0]);
    node->tmp = realloc(node->tmp, n);
    return 1;
}

scyte_node* scyte_dropout(scyte_node* x, scyte_node* dropout_rate)
{
    assert(dropout_rate->num_dims == 0);
    scyte_node* node = make_op2_node(DROPOUT, x, dropout_rate);
    node->forward = scyte_dropout_forward, node->backward = scyte_dropout_backward;
    if(!scyte_dropout_sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_dropout_forward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    int* keep_elements = (int*)node->tmp;
    float dropout_rate = scyte_is_const(operand) || scyte_is_var(operand)? 0.f : *node->children[1]->vals;
    float scale = 1.f / (1.f - dropout_rate);
    for(int i = 0; i < n; ++i) {
        int keep = random_uniform(0.f, 1.f) >= dropout_rate;
        node->vals[i] = keep ? operand->vals[i]*scale : 0.f; // scale by s to keep expected value
        if(keep_elements != NULL) keep_elements[i] = keep;
    }
}

void scyte_dropout_backward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int n = scyte_num_elements(operand);
    int* keep_elements = (int*)node->tmp;
    float dropout_rate = scyte_is_const(operand) || scyte_is_var(operand)? 0.f : *node->children[1]->vals;
    float scale = 1.f / (1.f - dropout_rate);
    if(scyte_has_gradient(operand)) {
        for(int i = 0; i < n; ++i) {
            if(keep_elements[i]) {
                operand->delta[i] += scale*node->delta[i];
            }
        }
    }
}
