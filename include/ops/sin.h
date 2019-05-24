#ifndef OP_SIN_H
#define OP_SIN_H

#include "scyte.h"

scyte_node* scyte_sin(scyte_node* x);

void scyte_sin_forward(scyte_node* node);
void scyte_sin_backward(scyte_node* node);

#endif
