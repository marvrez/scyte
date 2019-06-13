#include "ops/reduce_sum.h"

#include "op.h"
#include "logger.h"
#include "blas.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static inline int get_axis(scyte_node* node)
{
    return *(int*)node->params;
}

static inline void set_axis(scyte_node* node, int axis)
{
    int* axis_ptr = (int*)calloc(1, sizeof(int));
    *axis_ptr = axis;
    node->params = axis_ptr;
    node->params_size = sizeof(int);
}

static inline int sync_dims(scyte_node* node, int axis)
{
    scyte_node* operand = node->children[0];
    if(axis < 0 || axis >= operand->num_dims) {
        LOG_ERRORF("axis must be in range [0, %d]\n", operand->num_dims -1);
        return -1;
    }
    node->num_dims = operand->num_dims - 1;
    int j = 0;
    for (int i = 0; i < operand->num_dims; ++i) {
        if (i != axis) {
            node->shape[j++] = operand->shape[i];
        }
    }
    return 1;
}

scyte_node* scyte_reduce_sum(scyte_node* node, int axis)
{
    scyte_node* out = make_op1_node(REDUCE_SUM, node);
    out->forward = scyte_reduce_sum_forward, out->backward = scyte_reduce_sum_backward;
    set_axis(out, axis);
    if(!sync_dims(out, axis)) {
        free_op_node(out);
        return NULL;
    }
    return out;
}

void scyte_reduce_sum_forward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int axis = get_axis(node);
    int axis_shape = operand->shape[axis];
    int shape0, shape1;
    get_reduced_dimensions(operand, axis, &shape0, &shape1);

    set_cpu(scyte_num_elements(node), 0.f, node->vals);
    for(int i = 0; i < shape0; ++i) {
        int out_base_idx = i*shape1;
        for(int j = 0; j < axis_shape; ++j) {
            int in_base_idx = (i*axis_shape + j)*shape1;
            for(int k = 0; k < shape1; ++k) {
                node->vals[out_base_idx + k] += operand->vals[in_base_idx + k];
            }
        }

    }
}

void scyte_reduce_sum_backward(scyte_node* node)
{
    scyte_node* operand = node->children[0];
    int axis = get_axis(node);
    int axis_shape = operand->shape[axis];
    int shape0, shape1;
    get_reduced_dimensions(operand, axis, &shape0, &shape1);

    if(scyte_has_gradient(operand)) {
        for(int i = 0; i < shape0; ++i) {
            int in_base_idx = i*shape1;
            for(int j = 0; j < axis_shape; ++j) {
                int out_base_idx = (i*axis_shape + j)*shape1;
                for(int k = 0; k < shape1; ++k) {
                    operand->delta[out_base_idx + k] += node->delta[in_base_idx + k];
                }
            }

        }
    }
}
