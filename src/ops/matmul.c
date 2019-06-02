#include "ops/matmul.h"

#include "blas.h"
#include "op.h"

#include <stdio.h>


static inline void get_rows_cols(scyte_node* node, int* rows, int* cols)
{
    *rows = node->num_dims == 1 ? 1 : node->shape[0];
    *cols = scyte_num_elements(node) / *rows;
}

static inline int sync_dims(scyte_node* node)
{
    int num_rows_x, num_cols_x, num_rows_y, num_cols_y;
    get_rows_cols(node->children[0], &num_rows_x, &num_cols_x);
    get_rows_cols(node->children[1], &num_rows_y, &num_cols_y);

    if(num_cols_x != num_rows_y) {
        fprintf(stderr, "[scyte_matmul] dimensions (%d != %d) were not properly synced, returning NULL\n", num_cols_x, num_rows_y);
        return 0;
    }
    node->num_dims = 2;
    node->shape[0] = num_rows_x, node->shape[1] = num_cols_y;
    return 1;
}

scyte_node* scyte_matmul(scyte_node* x, scyte_node* y)
{
    scyte_node* node = make_op2_node(MATMUL, x, y);
    node->forward = scyte_matmul_forward, node->backward = scyte_matmul_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_matmul_forward(scyte_node* node)
{
    scyte_node* x = node->children[0], *y = node->children[1];
    int num_rows_x, num_cols_x, num_rows_y, num_cols_y;
    get_rows_cols(x, &num_rows_x, &num_cols_x);
    get_rows_cols(y, &num_rows_y, &num_cols_y);

    set_cpu(num_rows_x*num_cols_y, 0.f, node->vals);
    if(x->vals != NULL && y->vals != NULL) {
        gemm_cpu(0, 0, num_rows_x, num_cols_y, num_cols_x,
                1.f, x->vals, y->vals, 1.f, node->vals);
    }
}

void scyte_matmul_backward(scyte_node* node)
{
    scyte_node* x = node->children[0], *y = node->children[1];
    int num_rows_x, num_cols_x, num_rows_y, num_cols_y;
    get_rows_cols(x, &num_rows_x, &num_cols_x);
    get_rows_cols(y, &num_rows_y, &num_cols_y);

    if(scyte_has_gradient(x) && y->vals != NULL) {
        gemm_cpu(0, 1, num_rows_x, num_cols_x, num_cols_y,
                1.f, node->delta, y->vals, 1.f, x->delta);
    }
    if(scyte_has_gradient(y) && x->vals != NULL) {
        gemm_cpu(1, 0, num_rows_y, num_cols_y, num_rows_x,
                1.f, x->vals, node->delta, 1.f, y->delta);
    }
}
