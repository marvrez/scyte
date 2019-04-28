#include "scyte.h"

#include "blas.h"

#include <stdlib.h>
#include <assert.h>

void scyte_free_graph(int n, scyte_node** nodes)
{
    for(int i = 0; i < n; ++i) {
        scyte_node* node = nodes[i];
        free(node->in); free(node->out); free(node->delta);
        free(node->tmp); free(node->params);
        free(node->children); free(node);
    }
    free(nodes);
}

static inline void scyte_propagate_marks(int n, scyte_node** nodes)
{
    for(int i = n-1; i>= 0; --i) {
        scyte_node* node = nodes[i];
        if(node->mark > 0) {
            for(int j = 0; j < node->num_children; ++j) {
                node->children[j]->mark = (node->children[j]->mark == 0)
                                            ? 1 : node->children[j]->mark;
            }
        }
    }
}

const float* scyte_forward(int n, scyte_node** nodes, int to)
{
    int i;
    if(to < 0 || to >= n) to = n - 1;
    for(i = 0; i < n; ++i) nodes[i]->mark = (i == to);
    scyte_propagate_marks(n, nodes);
    for(i = 0; i < n; ++i) {
        scyte_node* node = nodes[i];
        if(node->num_children > 0 && node->mark > 0) {
            node->forward(node);
        }
    }
    return nodes[to]->out;
}

void scyte_backward(int n, scyte_node** nodes, int from)
{
    int i;
    if(from < 0 || from >= n) from = n - 1;
    assert(nodes[from]->num_dims == 0);

    // mark nodes where gradients should flow through
    for(i = 0; i < n; ++i) nodes[i]->mark = (i == from);
    scyte_propagate_marks(n, nodes);

    // set all relevant gradients to 0
    for(i = 0; i <= from; ++i) {
        scyte_node* node = nodes[i];
        if(node->delta && node->mark > 0) {
            set_cpu(scyte_num_elements(node), 0, node->delta);
        }
    }

    //backprop
    nodes[from]->delta[0] = 1.f; // derivative of output w.r.t output is 1
    for(i = from; i >= 0; --i) {
        scyte_node* node = nodes[i];
        if (node->num_children > 0 && node->mark > 0) {
            node->backward(node);
        }
    }
    for(i = 0; i <= from; ++i) nodes[i]->mark = 0;
}
