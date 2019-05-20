#ifndef EXP_H
#define EXP_H

#include "scyte.h"

scyte_node* scyte_exp(scyte_node* x);

void scyte_exp_forward(scyte_node* node);
void scyte_exp_backward(scyte_node* node);

#endif
