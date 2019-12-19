#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "scyte.h"

// layer normalization; applied to the last dimension
scyte_node* scyte_normalize(scyte_node* x);

int scyte_normalize_sync_dims(scyte_node* node);

void scyte_normalize_forward(scyte_node* node);
void scyte_normalize_backward(scyte_node* node);

#endif
