#include "network.h"

#include <stdlib.h>

// switch between forward and backward propagation mode
static inline void switch_propagation_mode(scyte_network* net, int is_backward)
{
    for(int i = 0; i < net->n; ++i) {
        scyte_node* node = net->nodes[i];
        if(node->op_type == SELECT && node->num_children == 2) {
            *(int*)node->params = !!is_backward;
        }
    }
}

scyte_network* scyte_make_network(scyte_node* cost_node)
{
    return NULL;
}

const float* scyte_predict_network(scyte_network* net, float* data)
{
    int out_idx = scyte_find_node(net, OUTPUT);
    if(out_idx < 0) return NULL;
    scyte_feed_net(net, PLACEHOLDER, &data);
    return scyte_forward(net->n, net->nodes, out_idx);
}

void scyte_free_network(scyte_network* net)
{
    if(!net) return;
    free(net->vals); free(net->deltas); free(net->consts);
    scyte_free_graph(net->n, net->nodes);
    free(net);
}
