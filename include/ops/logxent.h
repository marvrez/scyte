#ifndef LOGXENT_H
#define LOGXENT_H

#include "scyte.h"

// binary cross-entropy for (0,1)
scyte_node* scyte_logistic_x_entropy(scyte_node* truth, scyte_node* pred);

int scyte_logistic_x_entropy_sync_dims(scyte_node* node);

void scyte_logistic_x_entropy_forward(scyte_node* node);
void scyte_logistic_x_entropy_backward(scyte_node* node);

#endif
