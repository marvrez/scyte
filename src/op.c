#include "op.h"

#include <string.h>

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
    fprintf(stderr, "Couldn't find operation %s\n", s);
    return UNKNOWN;
}

action_func* scyte_get_op(scyte_op_type op_type)
{

}
