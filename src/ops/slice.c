#include "ops/slice.h"

#include "op.h"
#include "blas.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static inline void get_slice_params(scyte_node* node, int* axis, int* start, int* size)
{
    assert(node->params);
    int* axis_params = (int*)node->params;
    *axis   = axis_params[0];
    *start  = axis_params[1];
    *size   = axis_params[2];
}

// must be called after a node has been initalized
static inline void set_slice_params(scyte_node* node, int axis, int start, int size)
{
    int* slice_params = (int*)calloc(3, sizeof(int));
    slice_params[0] = axis, slice_params[1] = start, slice_params[2] = size;
    node->params = slice_params;
    node->params_size = sizeof(int)*3;
}

static inline int sync_dims(scyte_node* node, int axis, int start, int size)
{
    scyte_node* input = node->children[0];
    if(size <= 0 || start < 0 || size > input->shape[axis]) return 0;
    scyte_copy_shape(input, node);
    node->shape[axis] = size;
    return 1;
}

scyte_node* scyte_slice(scyte_node* x, int axis, int start, int size)
{
    assert(size > 0 && start >= 0);
    scyte_node* node = make_op1_node(SLICE, x);
    set_slice_params(node, axis, start, size);
    node->forward = scyte_slice_forward, node->backward = scyte_slice_backward;
    if(!sync_dims(node, axis, start, size)) {
        free_op_node(node);
        return NULL;
    }
    char* input_shape = get_shape_string(x->num_dims, x->shape);
    char* output_shape = get_shape_string(node->num_dims, node->shape);
    fprintf(stderr, "slice      axis=%d,start=%d,size=%d    %s -> %s\n", 
            axis, start, size, input_shape, output_shape);
    free(input_shape);
    free(output_shape);
            
    return node;
}

void scyte_slice_forward(scyte_node* node)
{
    scyte_node* child = node->children[0];
    int axis, start, size;
    get_slice_params(node,  &axis, &start, &size);
    assert(axis >= 0 && axis < child->num_dims && size > 0);

    int shape0 = 1, shape1 = 1;
    get_reduced_dimensions(node, axis, &shape0, &shape1);

    for(int i = 0; i < shape0; ++i) {
        copy_cpu(size*shape1, 
                &child->vals[(i*child->shape[axis] + start)*shape1], 
                &node->vals[i*node->shape[axis]*shape1]);
    }
}

void scyte_slice_backward(scyte_node* node)
{
    scyte_node* child = node->children[0];
    int axis, start, size;
    get_slice_params(node,  &axis, &start, &size);
    assert(axis >= 0 && axis < child->num_dims && size > 0);

    int shape0 = 1, shape1 = 1;
    get_reduced_dimensions(node, axis, &shape0, &shape1);

    for(int i = 0; i < shape0; ++i) {
        axpy_cpu(size*shape1, 1.f, 
                &node->delta[i*node->shape[axis]*shape1], 
                &child->delta[(i*child->shape[axis] + start)*shape1]);
    }
}
