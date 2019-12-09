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

scyte_node* scyte_layer_input(int n);
scyte_node* scyte_layer_connected(scyte_node* in, int num_outputs);
scyte_node* scyte_layer_dropout(scyte_node* in, float dropout_rate);
scyte_node* scyte_layer_layernorm(scyte_node* in);
scyte_node* scyte_layer_cost(scyte_node* in, int num_out, cost_type type);

#endif
