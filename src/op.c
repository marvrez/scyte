#include "op.h"

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
    return node;
}

scyte_node* make_op2_node(scyte_op_type type, scyte_node* x, scyte_node* y)
{
    scyte_node* node = make_op_node(type, 0, 2);
    node->children[0] = x, node->children[1] = y;
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
        case MATMUL: return "matmul";
        case AVG: return "avg";
        case DROPOUT: return "dropout";
        case MAX: return "max";
        case SOFTMAX: return "softmax";
        case EXP: return "exp";
        case UNKNOWN: default: break;
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
    if(strcmp(s, "matmul")) return MATMUL;
    if(strcmp(s, "avg")) return AVG;
    if(strcmp(s, "dropout")) return DROPOUT;
    if(strcmp(s, "max")) return MAX;
    if(strcmp(s, "softmax")) return SOFTMAX;
    if(strcmp(s, "exp")) return EXP;
    fprintf(stderr, "couldn't find operation %s\n", s);
    return UNKNOWN;
}

void scyte_validate_node(scyte_node* node)
{
    for(int i = 0; i < node->num_children; ++i) {
        if(scyte_has_gradient(node->children[i])) {
            node->type = VAR;
            break;
        }
    }
}
