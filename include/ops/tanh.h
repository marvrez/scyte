#ifndef OP_TANH_H
#define OP_TANH_H

#include "scyte.h"

scyte_node* scyte_tanh(scyte_node* x);

int scyte_tanh_sync_dims(scyte_node* node);

void scyte_tanh_forward(scyte_node* node);
void scyte_tanh_backward(scyte_node* node);

#endif
