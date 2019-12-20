#ifndef CONV2D_H
#define CONV2D_H

#include "scyte.h"

scyte_node* scyte_conv2d(scyte_node* x, scyte_node* w, int stride, int padding);

int scyte_conv2d_sync_dims(scyte_node* node);

void scyte_conv2d_forward(scyte_node* node);
void scyte_conv2d_backward(scyte_node* node);

#endif
