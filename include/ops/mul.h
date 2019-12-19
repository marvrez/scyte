#ifndef MUL_H
#define MUL_H

#include "scyte.h"

scyte_node* scyte_mul(scyte_node* x, scyte_node* y);

int scyte_mul_sync_dims(scyte_node* node);

void scyte_mul_forward(scyte_node* node);
void scyte_mul_backward(scyte_node* node);

#endif
