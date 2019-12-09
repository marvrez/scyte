#include "network.h"

#include <stdlib.h>

// switch between forward and backward propagation mode
static inline void switch_propagation_mode(scyte_network* net, int is_backward)
{
    for(int i = 0; i < net->n; ++i) {
        scyte_node* node = net->nodes[i];
        if(node->op_type == 12 && node->num_children == 2) {
            *(int*)node->params = !!is_backward;
        }
    }
}

void scyte_free_network(scyte_network* net)
{
    if(!net) return;
    free(net->vals); free(net->deltas); free(net->consts);
    scyte_free_graph(net->n, net->nodes);
    free(net);
}

