#ifndef REDUCE_MEAN_H
#define REDUCE_MEAN_H

#include "scyte.h"

scyte_node* scyte_reduce_mean(scyte_node* node, int axis);

void scyte_reduce_mean_forward(scyte_node* node);
void scyte_reduce_mean_backward(scyte_node* node);

#endif
