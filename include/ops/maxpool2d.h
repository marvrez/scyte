#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#include "scyte.h"

scyte_node* scyte_maxpool2d(scyte_node* x, int size, int stride, int padding);

int scyte_maxpool2d_sync_dims(scyte_node* node);

void scyte_maxpool2d_forward(scyte_node* node);
void scyte_maxpool2d_backward(scyte_node* node);

#endif
