#include "ops/categoricalxent.h"

#include "blas.h"
#include "logger.h"
#include "op.h"

#include <math.h>
#include <assert.h>
#include <stdio.h>

#define EPS 1e-9f

static inline int sync_dims(scyte_node* node, scyte_node* pred, scyte_node* truth)
{
    int n0 = scyte_num_elements(pred);
    int n1 = scyte_num_elements(truth);
    if(n0 != n1 || pred->shape[pred->num_dims - 1] != truth->shape[truth->num_dims - 1]) {
        LOG_ERRORF("dimensions (%d != %d) were not equal, returning NULL\n", n0, n1);
        return 0;
    }
    node->num_dims = 0;
    return 1;
}

scyte_node* scyte_categorical_x_entropy(scyte_node* truth, scyte_node* pred)
{
    scyte_node* node = make_op2_node(CATEGORICALXENT, pred, truth);
    node->forward = scyte_categorical_x_entropy_forward, node->backward = scyte_categorical_x_entropy_backward;
    if(!sync_dims(node, pred, truth)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_categorical_x_entropy_forward(scyte_node* node)
{
    scyte_node* pred = node->children[0], *truth = node->children[1];

    int dim = truth->shape[truth->num_dims - 1];
    int batch_size = scyte_num_elements(truth) / dim;
    float cost = 0.f;
    for(int i = 0; i < batch_size; ++i) {
        float* t = &truth->vals[i*dim];
        float* p = &pred->vals[i*dim];
        for(int j = 0; j < dim; ++j) {
            cost += t[j] ? -log(p[j] + EPS) : 0.f;
        }
    }
    node->vals[0] = cost / (float)batch_size;
}

void scyte_categorical_x_entropy_backward(scyte_node* node)
{
    scyte_node* pred = node->children[0], *truth = node->children[1];
    if(!scyte_has_gradient(pred)) return;
    int dim = truth->shape[truth->num_dims - 1];
    int batch_size = scyte_num_elements(truth) / dim;
    float s = node->delta[0] / (float)batch_size;
    for(int i = 0; i < batch_size; ++i) {
        float* t = &truth->vals[i*dim];
        float* p = &pred->vals[i*dim];
        float* dp = &pred->delta[i*dim];
        for(int j = 0; j < dim; ++j) {
            dp[j] += -t[j]*s/(p[j] + EPS);
        }
    }
}
