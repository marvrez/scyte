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
    scyte_propagate_gradient_mark(node);
    return node;
}

scyte_node* make_op2_node(scyte_op_type type, scyte_node* x, scyte_node* y)
{
    scyte_node* node = make_op_node(type, 0, 2);
    node->children[0] = x, node->children[1] = y;
    scyte_propagate_gradient_mark(node);
    return node;
}

scyte_node* make_opn_node(scyte_op_type type, int n, scyte_node** x)
{
    scyte_node* node = make_op_node(type, 0, n);
    for(int i = 0; i < n; ++i) node->children[i] = x[i];
    scyte_propagate_gradient_mark(node);
    return node;
}

void free_op_node(scyte_node* node)
{
    free(node->params);
    free(node->children);
    free(node);
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
        case L1_NORM: return "l1_norm";
        case MAXPOOL2D: return "maxpool2d";
        case CONV2D: return "conv2d";
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
    if(strcmp(s, "l1_norm")) return L1_NORM;
    if(strcmp(s, "maxpool2d")) return MAXPOOL2D;
    if(strcmp(s, "conv2d")) return CONV2D;
    LOG_ERRORF("couldn't find operation %s", s);
    return NOP;
}

void (*scyte_get_forward_function(scyte_op_type op_type)) (struct scyte_node*)
{
    switch(op_type) {
        case ADD: return scyte_add_forward;
        case SUB: return scyte_sub_forward;
        case MULTIPLY: return scyte_mul_forward;
        case SQUARE: return scyte_square_forward;
        case SIGMOID: return scyte_sigmoid_forward;
        case TANH: return scyte_tanh_forward;
        case RELU: return scyte_relu_forward;
        case CMATMUL: return scyte_cmatmul_forward;
        case MATMUL: return scyte_matmul_forward;
        case AVG: return scyte_avg_forward;
        case SELECT: return scyte_select_forward;
        case DROPOUT: return scyte_dropout_forward;
        case MAX: return scyte_max_forward;
        case SOFTMAX: return scyte_softmax_forward;
        case EXP: return scyte_exp_forward;
        case LOG: return scyte_log_forward;
        case SIN: return scyte_sin_forward;
        case MSE: return scyte_mse_forward;
        case RESHAPE: return scyte_reshape_forward;
        case CONCAT: return scyte_concat_forward;
        case SLICE: return scyte_slice_forward;
        case NORMALIZE: return scyte_normalize_forward;
        case REDUCE_SUM: return scyte_reduce_sum_forward;
        case REDUCE_MEAN: return scyte_reduce_mean_forward;
        case CATEGORICALXENT: return scyte_categorical_x_entropy_forward;
        case LOGXENT: return scyte_logistic_x_entropy_forward;
        case L1_NORM: return scyte_l1_norm_forward;
        case MAXPOOL2D: return scyte_maxpool2d_forward;
        case CONV2D: return scyte_conv2d_forward;
        case NOP: default: return NULL;
    }
    return NULL;
}

void (*scyte_get_backward_function(scyte_op_type op_type)) (struct scyte_node*)
{
    switch(op_type) {
        case ADD: return scyte_add_backward;
        case SUB: return scyte_sub_backward;
        case MULTIPLY: return scyte_mul_backward;
        case SQUARE: return scyte_square_backward;
        case SIGMOID: return scyte_sigmoid_backward;
        case TANH: return scyte_tanh_backward;
        case RELU: return scyte_relu_backward;
        case CMATMUL: return scyte_cmatmul_backward;
        case MATMUL: return scyte_matmul_backward;
        case AVG: return scyte_avg_backward;
        case SELECT: return scyte_select_backward;
        case DROPOUT: return scyte_dropout_backward;
        case MAX: return scyte_max_backward;
        case SOFTMAX: return scyte_softmax_backward;
        case EXP: return scyte_exp_backward;
        case LOG: return scyte_log_backward;
        case SIN: return scyte_sin_backward;
        case MSE: return scyte_mse_backward;
        case RESHAPE: return scyte_reshape_backward;
        case CONCAT: return scyte_concat_backward;
        case SLICE: return scyte_slice_backward;
        case NORMALIZE: return scyte_normalize_backward;
        case REDUCE_SUM: return scyte_reduce_sum_backward;
        case REDUCE_MEAN: return scyte_reduce_mean_backward;
        case CATEGORICALXENT: return scyte_categorical_x_entropy_backward;
        case LOGXENT: return scyte_logistic_x_entropy_backward;
        case L1_NORM: return scyte_l1_norm_backward;
        case MAXPOOL2D: return scyte_maxpool2d_backward;
        case CONV2D: return scyte_conv2d_backward;
        case NOP: default: return NULL;
    }
    return NULL;
}

int (*scyte_get_resync_function(scyte_op_type op_type)) (struct scyte_node*)
{
    switch(op_type) {
        case ADD: return scyte_add_sync_dims;
        case SUB: return scyte_sub_sync_dims;
        case MULTIPLY: return scyte_mul_sync_dims;
        case SQUARE: return scyte_square_sync_dims;
        case SIGMOID: return scyte_sigmoid_sync_dims;
        case TANH: return scyte_tanh_sync_dims;
        case RELU: return scyte_relu_sync_dims;
        case CMATMUL: return scyte_cmatmul_sync_dims;
        case MATMUL: return scyte_matmul_sync_dims;
        case AVG: return scyte_avg_sync_dims;
        case SELECT: return scyte_select_sync_dims;
        case DROPOUT: return scyte_dropout_sync_dims;
        case MAX: return scyte_max_sync_dims;
        case SOFTMAX: return scyte_softmax_sync_dims;
        case EXP: return scyte_exp_sync_dims;
        case LOG: return scyte_log_sync_dims;
        case SIN: return scyte_sin_sync_dims;
        case MSE: return scyte_mse_sync_dims;
        case RESHAPE: return scyte_reshape_sync_dims;
        case CONCAT: return scyte_concat_sync_dims;
        case SLICE: return scyte_slice_sync_dims;
        case NORMALIZE: return scyte_normalize_sync_dims;
        case REDUCE_SUM: return scyte_reduce_sum_sync_dims;
        case REDUCE_MEAN: return scyte_reduce_mean_sync_dims;
        case CATEGORICALXENT: return scyte_categorical_x_entropy_sync_dims;
        case LOGXENT: return scyte_logistic_x_entropy_sync_dims;
        case L1_NORM: return scyte_l1_norm_sync_dims;
        case MAXPOOL2D: return scyte_maxpool2d_sync_dims;
        case CONV2D: return scyte_conv2d_sync_dims;
        case NOP: default: return NULL;
    }
    return NULL;
}

void get_reduced_dimensions(scyte_node* node, int axis, int* shape0, int* shape1)
{
    *shape0 = *shape1 = 1;
    for(int i = 0; i < axis; ++i) *shape0 *= node->shape[i];
    for(int i = axis + 1; i < node->num_dims; ++i) *shape1 *= node->shape[i];
}

void scyte_propagate_gradient_mark(scyte_node* node)
{
    for(int i = 0; i < node->num_children; ++i) {
        if(scyte_has_gradient(node->children[i])) {
            node->type |= VAR;
            break;
        }
    }
}
