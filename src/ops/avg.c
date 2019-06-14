#include "ops/avg.h"

#include "logger.h"
#include "op.h"
#include "blas.h"

#include <stdio.h>
#include <assert.h>

static inline int sync_dims(scyte_node* node)
{
    int n = scyte_num_elements(node->children[0]);
    for(int i = 1; i < node->num_children; ++i) {
        int num_elements_children = scyte_num_elements(node->children[i]);
        if(num_elements_children != n) {
            LOG_ERRORF("dimensions %d != %d for child %d was not properly synced, returning NULL\n",
                i, n, num_elements_children);
            return 0;
        }
    }
    scyte_copy_shape(node->children[0], node);
    return 1;
}

scyte_node* scyte_avg(int n, scyte_node** nodes)
{
    scyte_node* node = make_opn_node(AVG, n, nodes);
    node->forward = scyte_avg_forward, node->backward = scyte_avg_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_avg_forward(scyte_node* node)
{
    scyte_node* child = node->children[0];
    int n = scyte_num_elements(child);
    assert(node->num_children > 0);
    float s = 1.f / (float) node->num_children;

    copy_cpu(n, child->vals, node->vals);
    for(int i = 1; i < node->num_children; ++i) {
        axpy_cpu(n, 1.f, node->children[i]->vals, node->vals);
    }
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        node->vals[i] *= s;
    }
}

void scyte_avg_backward(scyte_node* node)
{
    scyte_node* child = node->children[0];
    int n = scyte_num_elements(child);
    float s = 1.f / (float) node->num_children;
    for(int i = 0; i < n; ++i) {
        child = node->children[i];
        if(scyte_has_gradient(child)) {
            axpy_cpu(n, s, node->delta, child->delta);
        }
    }
}
