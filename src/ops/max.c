#include "ops/max.h"

#include "op.h"
#include "logger.h"
#include "blas.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int scyte_max_sync_dims(scyte_node* node)
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
    // node->tmp stores the index for the maximum value 
    // of the vals for a child
    int* max_val_idx = (int*)calloc(n, sizeof(int));
    node->tmp = max_val_idx;
    return 1;
}

scyte_node* scyte_max(int n, scyte_node** nodes)
{
    scyte_node* node = make_opn_node(MAX, n, nodes);
    node->forward = scyte_max_forward, node->backward = scyte_max_backward;
    if(!scyte_max_sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_max_forward(scyte_node* node)
{
    scyte_node* child = node->children[0];
    int n = scyte_num_elements(child);
    int* max_val_idx = (int*)node->tmp;

    memset(max_val_idx, 0, n*sizeof(int));
    copy_cpu(n, child->vals, node->vals);
    for(int i = 1; i < node->num_children; ++i) {
        child = node->children[i];
        for(int j = 0; j < n; ++j)  {
            if(child->vals[j] > node->vals[j]) {
                node->vals[j] = child->vals[j];
                max_val_idx[j] = i;
            }
        }
    }
}

void scyte_max_backward(scyte_node* node)
{
    scyte_node* child = node->children[0];
    int n = scyte_num_elements(child);
    int* max_val_idx = (int*)node->tmp;
    for(int i = 0; i < n; ++i) {
        child = node->children[max_val_idx[i]];
        if(scyte_has_gradient(child)) {
            child->delta[i] += node->delta[i];
        }
    }
}
