#ifndef SQUARE_H
#define SQUARE_H

#include "scyte.h"

scyte_node* scyte_square(scyte_node* x);

void scyte_square_forward(scyte_node* node);
void scyte_square_backward(scyte_node* node);

#endif
