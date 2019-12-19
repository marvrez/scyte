#include "ops/reshape.h"

#include "op.h"
#include "blas.h"
#include "utils.h"
#include "logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// must be called after a node has been initalized
static inline void set_shape(scyte_node* node, int n, int* shape)
{
    int* shape_params = NULL;
    if(n) {
        shape_params = (int*)calloc(n+1, sizeof(int));
        shape_params[0] = n;
        for(int i = 1; i < n; ++i) {
            shape_params[i] = shape ? shape[i] : -1;
        }
    }
    node->params = shape_params;
    node->params_size = sizeof(int)*(n+1);
}

static inline int* get_shape(scyte_node* node, int* n)
{
    int* shape_params = (int*)node->params;
    *n = shape_params[0];
    return &shape_params[1];
}

static inline int sync_dims(scyte_node* node)
{
    int n;
    int* shape = get_shape(node, &n);

    scyte_node* input = node->children[0];
    int num_elements_input = scyte_num_elements(input);
    if(shape == NULL || n <= 0) {
        scyte_copy_shape(input, node);
        return 1;
    }

    node->num_dims = n;
    int num_elements = 1, num_missing = 0;
    for(int i = 0; i < n; ++i) {
        node->shape[i] = shape[i];
        if(node->shape[i] <= 0) ++num_missing;
        else num_elements *= node->shape[i];
    }
    if(num_missing > 1) {
        LOG_ERROR("can only specify one unknown dimension\n");
        return 0;
    }
    if((num_missing == 0 && num_elements != num_elements_input) ||
        (num_missing == 1 && num_elements_input % num_elements != 0)) {
        char* shape_string = get_shape_string(n, shape);
        LOG_ERRORF("cannot reshape array of size %d into shape %s\n",
                num_elements_input, shape_string);
        free(shape_string);
        return 0;
    }
    for(int i = 0; i < n; ++i) {
        if(node->shape[i] <= 0) {
            node->shape[i] = num_elements_input / num_elements;
        }
    }
    return 1;
}

scyte_node* scyte_reshape(scyte_node* x, int n, int* shape)
{
    scyte_node* node = make_op1_node(RESHAPE, x);
    set_shape(node, n, shape);
    node->forward = scyte_reshape_forward, node->backward = scyte_reshape_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    char* input_shape = get_shape_string(x->num_dims, x->shape);
    char* output_shape = get_shape_string(node->num_dims, node->shape);
    fprintf(stderr, "reshape                             %s -> %s\n", 
            input_shape,
            output_shape
    );
    free(input_shape);
    free(output_shape);
            
    return node;
}

void scyte_reshape_forward(scyte_node* node)
{
    scyte_node* child = node->children[0];
    copy_cpu(scyte_num_elements(node), child->vals, node->vals);
}

void scyte_reshape_backward(scyte_node* node)
{
    scyte_node* child = node->children[0];
    axpy_cpu(scyte_num_elements(node), 1.f, node->delta, child->delta);
}
