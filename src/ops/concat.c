#include "ops/concat.h"

#include "op.h"
#include "blas.h"
#include "logger.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static inline void get_axis(scyte_node* node, int* axis)
{
    assert(node->params);
    int* axis_param = (int*)node->params;
    *axis = *axis_param;
}

// must be called after a node has been initalized
static inline void set_axis(scyte_node* node, int axis)
{
    int* axis_param = (int*)calloc(1, sizeof(int));
    *axis_param = axis;
    node->params = axis_param;
    node->params_size = sizeof(int);
}

static inline int sync_dims(scyte_node* node, int axis)
{
    scyte_node* input = node->children[0];
    for(int i = 1; i < node->num_children; ++i) {
        if(node->children[i]->num_dims != input->num_dims) {
            LOG_ERRORF("child %d and input doesn't have same num_dims (%d != %d)",
                    i, node->children[i]->num_dims, input->num_dims);
            return 0;
        }
        for(int j = 0; j < input->num_dims; ++j) {
            if (j != axis && input->shape[j] != node->children[i]->shape[j]){
                LOG_ERRORF("child %d and input doesn't have same dims (%d != %d)",
                        i, node->children[i]->shape[j], input->shape[j]);
                return 0;
            }
        }
    }
    scyte_copy_shape(input, node);
    node->shape[axis] = node->num_children*input->shape[axis];
    return 1;
}

scyte_node* scyte_concat(int axis, int n, scyte_node** x)
{
    scyte_node* node = make_opn_node(CONCAT, n, x);
    set_axis(node, axis);
    node->forward = scyte_concat_forward, node->backward = scyte_concat_backward;
    if(!sync_dims(node, axis)) {
        free_op_node(node);
        return NULL;
    }
    char* input_shape = get_shape_string(x[0]->num_dims, x[0]->shape);
    char* output_shape = get_shape_string(node->num_dims, node->shape);
    fprintf(stderr, "concat         axis=%d               %s -> %s\n",
            axis, 
            input_shape,
            output_shape
    );
    free(input_shape);
    free(output_shape);
            
    return node;
}

void scyte_concat_forward(scyte_node* node)
{
    scyte_node* child = node->children[0];
    int axis;
    get_axis(node,  &axis);

    int shape0 = 1, shape1 = 1;
    get_reduced_dimensions(node, axis, &shape0, &shape1);
    for(int i = 0; i < shape0; ++i) {
        for(int j = 0, k = 0; j < node->num_children; ++j) {
            child = node->children[j];
            copy_cpu(child->shape[axis]*shape1,
                    &child->vals[i*child->shape[axis]*shape1],
                    &node->vals[(i*node->shape[axis] + k)*shape1]);
            k += child->shape[axis];
        }
    }
}

void scyte_concat_backward(scyte_node* node)
{
    scyte_node* child = node->children[0];
    int axis;
    get_axis(node,  &axis);

    int shape0 = 1, shape1 = 1;
    get_reduced_dimensions(node, axis, &shape0, &shape1);
    for(int i = 0; i < shape0; ++i) {
        for(int j = 0, k = 0; j < node->num_children; ++j) {
            child = node->children[j];
            if(!scyte_has_gradient(child)) continue;
            axpy_cpu(child->shape[axis]*shape1, 1.f,
                    &node->delta[(i*node->shape[axis] + k)*shape1],
                    &child->delta[i*child->shape[axis]*shape1]);
            k += child->shape[axis];
        }
    }
}
