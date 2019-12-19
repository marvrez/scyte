#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "scyte.h"

scyte_node* scyte_softmax(scyte_node* x);

int scyte_softmax_sync_dims(scyte_node* node);

void scyte_softmax_forward(scyte_node* node);
void scyte_softmax_backward(scyte_node* node);

#endif
