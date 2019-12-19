#ifndef RELU_H
#define RELU_H

#include "scyte.h"

scyte_node* scyte_relu(scyte_node* x);

int scyte_relu_sync_dims(scyte_node* node);

void scyte_relu_forward(scyte_node* node);
void scyte_relu_backward(scyte_node* node);

#endif
