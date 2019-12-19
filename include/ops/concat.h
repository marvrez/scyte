#ifndef CONCAT_H
#define CONCAT_H

#include "scyte.h"

scyte_node* scyte_concat(int axis, int n, scyte_node** x);

int scyte_concat_sync_dims(scyte_node* node);

void scyte_concat_forward(scyte_node* node);
void scyte_concat_backward(scyte_node* node);

#endif
