#ifndef OP_H
#define OP_H

#include "scyte.h"

typedef void (*action_func)(struct scyte_node*);

char* scyte_get_op_string(scyte_op_type op_type);
scyte_op_type scyte_get_op_type(char* s);
action_func* scyte_get_op(scyte_op_type op_type);

#endif
