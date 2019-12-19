#ifndef OP_MAX_H
#define OP_MAX_H

#include "scyte.h"

scyte_node* scyte_max(int n, scyte_node** nodes);

int scyte_max_sync_dims(scyte_node* node);

void scyte_max_forward(scyte_node* node);
void scyte_max_backward(scyte_node* node);

#endif
