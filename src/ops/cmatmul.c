#include "ops/cmatmul.h"

#include "blas.h"
#include "op.h"

#include <stdio.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static inline void get_rows_cols(scyte_node* node, int num_cols, int* rows, int* cols)
{
    *cols = 1;
    for (int i = node->num_dims - 1; i >= 0; --i) {
        if(*cols < num_cols) *cols *= node->shape[i];
    }
    *rows = scyte_num_elements(node) / *cols;
}

static inline int sync_dims(scyte_node* node)
{
    scyte_node* x = node->children[0], *y = node->children[1];

    int num_rows_x, num_cols_x, num_rows_y, num_cols_y;
    int num_cols = MAX(x->shape[x->num_dims-1], y->shape[y->num_dims - 1]);
    get_rows_cols(x, num_cols, &num_rows_x, &num_cols_x);
    get_rows_cols(y, num_cols, &num_rows_y, &num_cols_y);

    if(num_cols_x != num_cols_y) {
        fprintf(stderr, "[scyte_cmatmul] dimensions (%d != %d) were not properly synced, returning NULL\n", num_cols_x, num_cols_y);
        return 0;
    }
    node->num_dims = 2;
    node->shape[0] = num_rows_x, node->shape[1] = num_rows_y;
    return 1;
}

scyte_node* scyte_cmatmul(scyte_node* x, scyte_node* y)
{
    scyte_node* node = make_op2_node(CMATMUL, x, y);
    node->forward = scyte_cmatmul_forward, node->backward = scyte_cmatmul_backward;
    if(!sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_cmatmul_forward(scyte_node* node)
{
    scyte_node* x = node->children[0], *y = node->children[1];

    int num_rows_x, num_cols_x, num_rows_y, num_cols_y;
    int num_cols = MAX(x->shape[x->num_dims-1], y->shape[y->num_dims - 1]);
    get_rows_cols(x, num_cols, &num_rows_x, &num_cols_x);
    get_rows_cols(y, num_cols, &num_rows_y, &num_cols_y);

    set_cpu(num_rows_x*num_rows_y, 0.f, node->vals);
    if(x->vals != NULL && y->vals != NULL) {
        gemm_cpu(0, 1, num_rows_x, num_rows_y, num_cols,
                1.f, x->vals, y->vals, 1.f, node->vals);
    }
}

void scyte_cmatmul_backward(scyte_node* node)
{
    scyte_node* x = node->children[0], *y = node->children[1];

    int num_rows_x, num_cols_x, num_rows_y, num_cols_y;
    int num_cols = MAX(x->shape[x->num_dims-1], y->shape[y->num_dims - 1]);
    get_rows_cols(x, num_cols, &num_rows_x, &num_cols_x);
    get_rows_cols(y, num_cols, &num_rows_y, &num_cols_y);

    if(scyte_has_gradient(x) && y->vals != NULL) {
        gemm_cpu(0, 0, num_rows_x, num_cols, num_rows_y,
                1.f, node->delta, y->vals, 1.f, x->delta);
    }
    if(scyte_has_gradient(y) && x->vals != NULL) {
        gemm_cpu(1, 0, num_rows_y, num_cols, num_rows_x,
                1.f, node->delta, x->vals, 1.f, y->delta);
    }
}
