#ifndef OP_H
#define OP_H

#include "scyte.h"

#include "ops/add.h"
#include "ops/sub.h"
#include "ops/square.h"
#include "ops/exp.h"
#include "ops/log.h"
#include "ops/relu.h"
#include "ops/sigmoid.h"
#include "ops/tanh.h"
#include "ops/softmax.h"
#include "ops/dropout.h"
#include "ops/sin.h"
#include "ops/mul.h"
#include "ops/mse.h"
#include "ops/matmul.h"
#include "ops/cmatmul.h"
#include "ops/max.h"
#include "ops/avg.h"
#include "ops/select.h"
#include "ops/reduce_sum.h"
#include "ops/reduce_mean.h"
#include "ops/slice.h"
#include "ops/concat.h"
#include "ops/reshape.h"
#include "ops/logxent.h"
#include "ops/categoricalxent.h"
#include "ops/normalize.h"

scyte_node* make_op_node(scyte_op_type type, int num_dims, int num_children);
scyte_node* make_op1_node(scyte_op_type type, scyte_node* x);
scyte_node* make_op2_node(scyte_op_type type, scyte_node* x, scyte_node* y);
scyte_node* make_opn_node(scyte_op_type type, int n, scyte_node** x);
void free_op_node(scyte_node* node);

char* scyte_get_op_string(scyte_op_type op_type);
scyte_op_type scyte_get_op_type(char* s);

// checks if gradients can flow through the op-node, if so set type to VAR
void scyte_validate_node(scyte_node* node);

#endif
