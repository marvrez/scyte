#ifndef SIGMOID_H
#define SIGMOID_H

#include "scyte.h"

scyte_node* scyte_sigmoid(scyte_node* x);

int scyte_sigmoid_sync_dims(scyte_node* node);

void scyte_sigmoid_forward(scyte_node* node);
void scyte_sigmoid_backward(scyte_node* node);

#endif
