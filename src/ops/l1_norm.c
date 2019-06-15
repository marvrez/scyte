#include "ops/l1_norm.h"

#include "op.h"
#include "logger.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>

static inline int sync_dims(scyte_node* node)
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

scyte_node* scyte_l1_norm(scyte_node* truth, scyte_node* pred)
{
    scyte_node* node = make_op2_node(L1_NORM, truth, pred);
    node->forward = scyte_l1_norm_forward, node->backward = scyte_l1_norm_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_l1_norm_forward(scyte_node* node)
{
    scyte_node* truth = node->children[0], *pred = node->children[1];
    int n = scyte_num_elements(truth);
    float abs_diffs = 0.f; // sum of absolute differences
    #pragma omp parallel for reduction(+:abs_diffs)
    for(int i = 0; i < n; ++i) {
        abs_diffs += fabsf(truth->vals[i] - pred->vals[i]);
    }
    node->vals[0] = abs_diffs / (float)n;
}

void scyte_l1_norm_backward(scyte_node* node)
{
    scyte_node* truth = node->children[0], *pred = node->children[1];
    int n = scyte_num_elements(truth);
    if(scyte_has_gradient(pred)) {
        float s = node->delta[0] / n;
        #pragma omp parallel for
        for(int i = 0; i < n; ++i) {
            pred->delta[i] += s*get_sign(truth->vals[i] - pred->vals[i]);
        }
    }
}
