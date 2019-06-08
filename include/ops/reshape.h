#ifndef RESHAPE_H
#define RESHAPE_H

#include "scyte.h"

scyte_node* scyte_reshape(scyte_node* x, int n, int* shape);

void scyte_reshape_forward(scyte_node* node);
void scyte_reshape_backward(scyte_node* node);

#endif
