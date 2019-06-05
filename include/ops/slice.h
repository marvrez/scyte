#ifndef SLICE_H
#define SLICE_H

#include "scyte.h"

// slice an axis in inclusive range [start, start+size)
scyte_node* scyte_slice(scyte_node* x, int axis, int start, int size);

void scyte_slice_forward(scyte_node* node);
void scyte_slice_backward(scyte_node* node);

#endif
