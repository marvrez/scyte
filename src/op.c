#include "op.h"

#include "logger.h"

#include <string.h>
#include <stdlib.h>

scyte_node* make_op_node(scyte_op_type type, int num_dims, int num_children)
{
    scyte_node* node;
    if (num_dims >= SCYTE_MAX_DIMS) return NULL;
    node = (scyte_node*)calloc(1, sizeof(scyte_node));
    node->num_dims = num_dims, node->op_type = type, node->num_children = num_children;
    if (node->num_children) {
        node->children = (scyte_node**)calloc(node->num_children, sizeof(scyte_node*));
    }
    return node;
}

scyte_node* make_op1_node(scyte_op_type type, scyte_node* x)
{
    scyte_node* node = make_op_node(type, 0, 1);
    node->children[0] = x;
    scyte_validate_node(node);
    return node;
}

scyte_node* make_op2_node(scyte_op_type type, scyte_node* x, scyte_node* y)
{
    scyte_node* node = make_op_node(type, 0, 2);
    node->children[0] = x, node->children[1] = y;
    scyte_validate_node(node);
    return node;
}

scyte_node* make_opn_node(scyte_op_type type, int n, scyte_node** x)
{
    scyte_node* node = make_op_node(type, 0, n);
    for(int i = 0; i < n; ++i) node->children[i] = x[i];
    scyte_validate_node(node);
    return node;
}

void free_op_node(scyte_node* node)
{
    free(node->params);
    free(node->children); free(node);
}

char* scyte_get_op_string(scyte_op_type op_type)
{
    switch(op_type) {
        case ADD: return "add";
        case SUB: return "sub";
        case MULTIPLY: return "multiply";
        case SQUARE: return "square";
        case SIGMOID: return "sigmoid";
        case TANH: return "tanh";
        case RELU: return "relu";
        case CMATMUL: return "cmatmul";
        case MATMUL: return "matmul";
        case AVG: return "avg";
        case SELECT: return "select";
        case DROPOUT: return "dropout";
        case MAX: return "max";
        case SOFTMAX: return "softmax";
        case EXP: return "exp";
        case LOG: return "log";
        case SIN: return "sin";
        case MSE: return "mse";
        case RESHAPE: return "reshape";
        case CONCAT: return "concat";
        case SLICE: return "slice";
        case NORMALIZE: return "normalize";
        case REDUCE_SUM: return "reduce_sum";
        case REDUCE_MEAN: return "reduce_mean";
        case CATEGORICALXENT: return "categoricalxent";
        case LOGXENT: return "logxent";
        case NOP: default: break;
    }
    return "unknown";
}

scyte_op_type scyte_get_op_type(char* s)
{
    if(strcmp(s, "add")) return ADD;
    if(strcmp(s, "sub")) return SUB;
    if(strcmp(s, "multiply")) return MULTIPLY;
    if(strcmp(s, "square")) return SQUARE;
    if(strcmp(s, "sigmoid")) return SIGMOID;
    if(strcmp(s, "tanh")) return TANH;
    if(strcmp(s, "relu")) return RELU;
    if(strcmp(s, "cmatmul")) return CMATMUL;
    if(strcmp(s, "matmul")) return MATMUL;
    if(strcmp(s, "avg")) return AVG;
    if(strcmp(s, "select")) return SELECT;
    if(strcmp(s, "dropout")) return DROPOUT;
    if(strcmp(s, "max")) return MAX;
    if(strcmp(s, "softmax")) return SOFTMAX;
    if(strcmp(s, "exp")) return EXP;
    if(strcmp(s, "log")) return LOG;
    if(strcmp(s, "sin")) return SIN;
    if(strcmp(s, "mse")) return MSE;
    if(strcmp(s, "reshape")) return RESHAPE;
    if(strcmp(s, "concat")) return CONCAT;
    if(strcmp(s, "slice")) return SLICE;
    if(strcmp(s, "normalize")) return NORMALIZE;
    if(strcmp(s, "reduce_sum")) return REDUCE_SUM;
    if(strcmp(s, "reduce_mean")) return REDUCE_MEAN;
    if(strcmp(s, "categoricalxent")) return CATEGORICALXENT;
    if(strcmp(s, "logxent")) return LOGXENT;
    LOG_ERRORF("couldn't find operation %s", s);
    return NOP;
}

void get_reduced_dimensions(scyte_node* node, int axis, int* shape0, int* shape1)
{
    *shape0 = *shape1 = 1;
    for(int i = 0; i < axis; ++i) *shape0 *= node->shape[i];
    for(int i = axis + 1; i < node->num_dims; ++i) *shape1 *= node->shape[i];
}

void scyte_validate_node(scyte_node* node)
{
    for(int i = 0; i < node->num_children; ++i) {
        if(scyte_has_gradient(node->children[i])) {
            node->type |= VAR;
            break;
        }
    }
}
