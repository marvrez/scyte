#ifndef REDUCE_SUM_H
#define REDUCE_SUM_H

#include "scyte.h"

scyte_node* scyte_reduce_sum(scyte_node* node, int axis);

void scyte_reduce_sum_forward(scyte_node* node);
void scyte_reduce_sum_backward(scyte_node* node);

#endif
