#ifndef LAYERS_H
#define LAYERS_H

#include "scyte.h"

typedef enum {
    COST_BINARY_CROSS_ENTROPY,
    COST_CROSS_ENTROPY,
    COST_L1,
    COST_L2,
    //COST_HUBER,
} cost_type;

const char* get_cost_string(cost_type type);

// Generates a network from a computational graph.
// A network must have at least one scalar cost node (i.e. whose num_dims==0).
scyte_network* scyte_make_network(scyte_node* cost_node);
void scyte_free_network(scyte_network* net);

void scyte_save_network(const char* filename, scyte_network* net);
scyte_network* scyte_load_network(const char* filename);

scyte_node* scyte_layer_input(int n);
scyte_node* scyte_layer_connected(scyte_node* in, int num_outputs);
scyte_node* scyte_layer_dropout(scyte_node* in, float dropout_rate);
scyte_node* scyte_layer_layernorm(scyte_node* in);
scyte_node* scyte_layer_cost(scyte_node* in, int num_out, cost_type type);

#endif
