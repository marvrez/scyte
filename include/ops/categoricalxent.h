#ifndef CATEGORICALXENT_H
#define CATEGORICALXENT_H

#include "scyte.h"

// multi-class cross-entropy; pred is the prediction and truth is the truth
scyte_node* scyte_categorical_x_entropy(scyte_node* truth, scyte_node* pred);

void scyte_categorical_x_entropy_forward(scyte_node* node);
void scyte_categorical_x_entropy_backward(scyte_node* node);

#endif
