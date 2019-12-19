#ifndef DROPOUT_H
#define DROPOUT_H

#include "scyte.h"

scyte_node* scyte_dropout(scyte_node* x, scyte_node* dropout_rate);

int scyte_dropout_sync_dims(scyte_node* node);

void scyte_dropout_forward(scyte_node* node);
void scyte_dropout_backward(scyte_node* node);

#endif
