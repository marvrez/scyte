#ifndef OP_AVG_H
#define OP_AVG_H

#include "scyte.h"

scyte_node* scyte_avg(int n, scyte_node** nodes);

void scyte_avg_forward(scyte_node* node);
void scyte_avg_backward(scyte_node* node);

#endif
