#ifndef ABSOLUTE_DIFFERENCES_H
#define ABSOLUTE_DIFFERENCES_H

#include "scyte.h"

scyte_node* scyte_l1_norm(scyte_node* truth, scyte_node* pred);

void scyte_l1_norm_forward(scyte_node* node);
void scyte_l1_norm_backward(scyte_node* node);

#endif
