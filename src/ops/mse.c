#include "ops/mse.h"

#include "op.h"
#include "logger.h"

#include <stdio.h>

int scyte_mse_sync_dims(scyte_node* node)
{
    int n0 = scyte_num_elements(node->children[0]);
    int n1 = scyte_num_elements(node->children[1]);
    if(n0 != n1) {
        LOG_ERRORF("dimensions (%d != %d) were not properly synced, returning NULL\n", n0, n1);
        return 0;
    }
    node->num_dims = 0;
    return 1;
}

scyte_node* scyte_mse(scyte_node* truth, scyte_node* pred)
{
    scyte_node* node = make_op2_node(MSE, truth, pred);
    node->forward = scyte_mse_forward, node->backward = scyte_mse_backward;
    if(!scyte_mse_sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_mse_forward(scyte_node* node)
{
    scyte_node* truth = node->children[0], *pred = node->children[1];
    int n = scyte_num_elements(truth);
    float sse = 0.f; // sum of squared errors
    for(int i = 0; i < n; ++i) {
        float diff = truth->vals[i] - pred->vals[i];
        sse += diff*diff;
    }
    node->vals[0] = sse / (float)n;
}

void scyte_mse_backward(scyte_node* node)
{
    scyte_node* truth = node->children[0], *pred = node->children[1];
    int n = scyte_num_elements(truth);
    if(scyte_has_gradient(pred)) {
        float s = 2.f * node->delta[0] / n;
        for(int i = 0; i < n; ++i) {
            pred->delta[i] += s*(pred->vals[i] - truth->vals[i]);
        }
    }
}
