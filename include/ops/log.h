#ifndef OP_LOG_H
#define OP_LOG_H

#include "scyte.h"

scyte_node* scyte_log(scyte_node* x);

int scyte_log_sync_dims(scyte_node* node);

void scyte_log_forward(scyte_node* node);
void scyte_log_backward(scyte_node* node);

#endif
