#include "ops/select.h"

#include "logger.h"
#include "op.h"
#include "blas.h"

#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <assert.h>

static inline int get_node_idx(scyte_node* node)
{
    int node_idx = *(int*)node->params;
    if(node_idx < 0) node_idx += node->num_children;
    assert(node_idx >= 0 && node_idx < node->num_children);
    return node_idx;
}

// must be called after a node has been initalized
static inline void set_node_idx(scyte_node* node, int node_idx)
{
    if(node_idx < 0) node_idx += node->num_children;
    assert(node_idx >= 0 && node_idx < node->num_children);
    int* node_idx_ptr = (int*)calloc(1, sizeof(int));
    *node_idx_ptr = node_idx;
    node->params = node_idx_ptr;
    node->params_size = sizeof(int);
}

static inline int sync_dims(scyte_node* node)
{
    int node_idx = get_node_idx(node);
    scyte_node* chosen_node = node->children[node_idx];
    int n = scyte_num_elements(chosen_node);
    for(int i = 0; i < node->num_children; ++i) {
        scyte_node* child = node->children[i];
        int num_elements_children = scyte_num_elements(child);
        if(child->num_dims != chosen_node->num_dims || num_elements_children != n) {
            LOG_ERRORF("nun_elements (%d != %d) or num_dims (%d != %d) for child %d was not properly synced, returning NULL\n",
                n, num_elements_children, child->num_dims, chosen_node->num_dims, i);
            return 0;
        }
    }
    scyte_copy_shape(chosen_node, node);
    return 1;
}

scyte_node* scyte_select(int node_idx, int n, scyte_node** nodes)
{
    scyte_node* node = make_opn_node(SELECT, n, nodes);
    set_node_idx(node, node_idx);
    node->forward = scyte_select_forward, node->backward = scyte_select_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    fprintf(stderr, "select         idx=%d\n", node_idx);
    return node;
}

scyte_node* scyte_dynamic_select(int n, scyte_node** nodes)
{
    scyte_node* node = scyte_select(0, n, nodes);
    return node;
}

void scyte_select_forward(scyte_node* node)
{
    int node_idx = get_node_idx(node);
    scyte_node* chosen_node = node->children[node_idx];
    int n = scyte_num_elements(chosen_node);
    copy_cpu(n, chosen_node->vals, node->vals);
}

void scyte_select_backward(scyte_node* node)
{
    int node_idx = get_node_idx(node);
    scyte_node* chosen_node = node->children[node_idx];
    int n = scyte_num_elements(chosen_node);
    if(scyte_has_gradient(chosen_node)) {
        axpy_cpu(n, 1.f, node->delta, chosen_node->delta);
    }
}
