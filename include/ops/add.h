#ifndef ADD_H
#define ADD_H

#include "scyte.h"

scyte_node* scyte_add(scyte_node* x, scyte_node* y);

int scyte_add_sync_dims(scyte_node* node);

void scyte_add_forward(scyte_node* node);
void scyte_add_backward(scyte_node* node);

#endif
