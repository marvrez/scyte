#ifndef OP_SELECT_H
#define OP_SELECT_H

#include "scyte.h"

// select node by node index
scyte_node* scyte_select(int n, scyte_node** nodes, int node_idx);
// utility to switch between two nodes during runtime
scyte_node* scyte_dynamic_select(int n, scyte_node** nodes);

void scyte_select_forward(scyte_node* node);
void scyte_select_backward(scyte_node* node);

#endif
