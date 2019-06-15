#ifndef MSE_H
#define MSE_H

#include "scyte.h"

scyte_node* scyte_mse(scyte_node* truth, scyte_node* pred);

void scyte_mse_forward(scyte_node* node);
void scyte_mse_backward(scyte_node* node);

#endif
