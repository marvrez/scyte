#ifndef CMATMUL_H
#define CMATMUL_H

#include "scyte.h"

// this one transposes y, meaning: Z = X * Y^T
scyte_node* scyte_cmatmul(scyte_node* x, scyte_node* y);

int scyte_cmatmul_sync_dims(scyte_node* node);

void scyte_cmatmul_forward(scyte_node* node);
void scyte_cmatmul_backward(scyte_node* node);

#endif
