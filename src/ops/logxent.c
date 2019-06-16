#include "ops/logxent.h"

#include "blas.h"
#include "logger.h"
#include "op.h"

#include <math.h>
#include <assert.h>
#include <stdio.h>

#define EPS 1e-9

static inline int sync_dims(scyte_node* node)
{
    int n0 = scyte_num_elements(node->children[0]);
    int n1 = scyte_num_elements(node->children[1]);
    if(n0 != n1) {
        LOG_ERRORF("dimensions (%d != %d) were not equal, returning NULL\n", n0, n1);
        return 0;
    }
    node->num_dims = 0;
    return 1;
}

scyte_node* scyte_logistic_x_entropy(scyte_node* truth, scyte_node* pred)
{
    scyte_node* node = make_op2_node(LOGXENT, pred, truth);
    node->forward = scyte_logistic_x_entropy_forward, node->backward = scyte_logistic_x_entropy_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_logistic_x_entropy_forward(scyte_node* node)
{
    scyte_node* pred = node->children[0], *truth = node->children[1];
    int n = scyte_num_elements(pred);
    float cost = 0.f;
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        float p = pred->vals[i];
        float t = truth->vals[i];
        cost += -t*logf(p) - (1-t)*logf(1-p);
    }
    node->vals[0] = cost / (float)n;
}

void scyte_logistic_x_entropy_backward(scyte_node* node)
{
    scyte_node* pred = node->children[0], *truth = node->children[1];
    if(!scyte_has_gradient(pred)) return;
    int n = scyte_num_elements(pred);
    float s = node->delta[0] / (float)n;
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        float p = pred->vals[i];
        float t = truth->vals[i];
        pred->delta[i] += s*(-t/p + (1.f-t)/(1.f-p));
    }
}
