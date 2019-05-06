#ifndef OP_H
#define OP_H

#include "scyte.h"

#include "ops/add.h"

typedef void (*action_func)(struct scyte_node*);

scyte_node* make_op_node(scyte_op_type type, int num_dims, int num_children);
scyte_node* make_op1_node(scyte_op_type type, scyte_node* x);
scyte_node* make_op2_node(scyte_op_type type, scyte_node* x, scyte_node* y);

char* scyte_get_op_string(scyte_op_type op_type);
scyte_op_type scyte_get_op_type(char* s);

#endif
