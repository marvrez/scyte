#ifndef SUB_H
#define SUB_H

#include "scyte.h"

scyte_node* scyte_sub(scyte_node* x, scyte_node* y);

void scyte_sub_forward(scyte_node* node);
void scyte_sub_backward(scyte_node* node);

#endif
