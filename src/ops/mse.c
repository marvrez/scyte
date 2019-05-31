#include "ops/mse.h"

#include "op.h"

#include <stdio.h>

static inline int sync_dims(scyte_node* node)
{
    int n0 = scyte_num_elements(node->children[0]);
    int n1 = scyte_num_elements(node->children[1]);
    if(n0 != n1) {
        fprintf(stderr, "[scyte_mse] dimensions (%d != %d) were not properly synced, returning NULL\n", n0, n1);
        return 0;
    }
    node->num_dims = 0;
    return 1;
}

scyte_node* scyte_mse(scyte_node* y, scyte_node* y_hat)
{
    scyte_node* node = make_op2_node(MSE, y, y_hat);
    node->forward = scyte_mse_forward, node->backward = scyte_mse_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_mse_forward(scyte_node* node)
{
    scyte_node* y = node->children[0], *y_hat = node->children[1];
    int n = scyte_num_elements(y);
    float sse = 0.f; // sum of squared errors
    for(int i = 0; i < n; ++i) {
        sse += (y->vals[i] - y_hat->vals[i])*(y->vals[i] - y_hat->vals[i]);
    }
    node->vals[0] = sse / (float)n;
}

void scyte_mse_backward(scyte_node* node)
{
    scyte_node* y = node->children[0], *y_hat = node->children[1];
    int n = scyte_num_elements(y);
    if(scyte_has_gradient(y_hat)) {
        float s = 2.f * node->delta[0] / n;
        for(int i = 0; i < n; ++i) {
            y_hat->delta[i] += s*(y->vals[i] - y_hat->vals[i]);
        }
    }
}
