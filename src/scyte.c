#include "scyte.h"
#include "op.h"

#include "blas.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

static inline scyte_node* scyte_make_node(scyte_node_type type, unsigned num_dims, int shape[SCYTE_MAX_DIMS], float fill_val)
{
    scyte_node* node;
    if(num_dims > SCYTE_MAX_DIMS) return NULL;
    node = (scyte_node*)calloc(1, sizeof(scyte_node));
    node->num_dims = num_dims, node->type = type;
    memcpy(node->shape, shape, num_dims*sizeof(int));
    if(type != INPUT) {
        int num_elements = scyte_num_elements(node);
        node->vals = (float*)calloc(num_elements, sizeof(float));
        if (node->num_dims <= 1) set_cpu(num_elements, fill_val, node->vals);
        else {
            float s = 2.f / sqrtf((float)num_elements / node->shape[0]); // s = 2 / sqrt(n_in)
            for(int i = 0; i < num_elements; ++i) node->vals[i] = s*random_uniform(-1, 1);
        }
    }
    return node;
}

scyte_node* scyte_input(unsigned num_dims, int shape[SCYTE_MAX_DIMS])
{
    return scyte_make_node(INPUT, num_dims, shape, 0);
}

scyte_node* scyte_const(unsigned num_dims, int shape[SCYTE_MAX_DIMS], float fill_val)
{
    return scyte_make_node(CONST, num_dims, shape, fill_val);
}

scyte_node* scyte_var(unsigned num_dims, int shape[SCYTE_MAX_DIMS], float fill_val)
{
    return scyte_make_node(VAR, num_dims, shape, fill_val);
}

scyte_node** scyte_make_graph(int* num_nodes, int num_roots, scyte_node** roots)
{
    return NULL;
}

void scyte_free_graph(int n, scyte_node** nodes)
{
    for(int i = 0; i < n; ++i) {
        scyte_node* node = nodes[i];
        free(node->vals);free(node->delta);
        free(node->tmp); free(node->params);
        free(node->children); free(node);
    }
    free(nodes);
}

void scyte_copy_shape(const scyte_node* src, scyte_node* dst)
{
    dst->num_dims = src->num_dims;
    if(src->num_dims){
        memcpy(dst->shape, src->shape, src->num_dims*sizeof(int));
    }
}

static inline void scyte_propagate_marks(int n, scyte_node** nodes)
{
    for(int i = n-1; i>= 0; --i) {
        scyte_node* node = nodes[i];
        if(node->mark > 0) {
            for(int j = 0; j < node->num_children; ++j) {
                node->children[j]->mark = (node->children[j]->mark == 0)
                                            ? 1 : node->children[j]->mark;
            }
        }
    }
}

const float* scyte_forward(int n, scyte_node** nodes, int to)
{
    int i;
    if(to < 0 || to >= n) to = n - 1;
    for(i = 0; i < n; ++i) nodes[i]->mark = (i == to);
    scyte_propagate_marks(n, nodes);
    for(i = 0; i < n; ++i) {
        scyte_node* node = nodes[i];
        if(node->num_children > 0 && node->mark > 0) {
            node->forward(node);
        }
    }
    return nodes[to]->vals;
}

void scyte_backward(int n, scyte_node** nodes, int from)
{
    int i;
    if(from < 0 || from >= n) from = n - 1;
    assert(nodes[from]->num_dims == 0);

    // mark nodes where gradients should flow through
    for(i = 0; i < n; ++i) nodes[i]->mark = (i == from);
    scyte_propagate_marks(n, nodes);

    // set all relevant gradients to 0
    for(i = 0; i <= from; ++i) {
        scyte_node* node = nodes[i];
        if(node->delta && node->mark > 0) {
            set_cpu(scyte_num_elements(node), 0, node->delta);
        }
    }

    //backprop
    nodes[from]->delta[0] = 1.f; // derivative of output w.r.t output is 1
    for(i = from; i >= 0; --i) {
        scyte_node* node = nodes[i];
        if (node->num_children > 0 && node->mark > 0) {
            node->backward(node);
        }
    }
    for(i = 0; i <= from; ++i) nodes[i]->mark = 0;
}

void scyte_print_graph(int n, scyte_node** nodes)
{
    int i, j;
    for(i = 0; i < n; ++i) nodes[i]->mark = i;
    printf("$node\tshape\t\ttype\n");
    printf("----------------------------\n");
    for(i = 0; i < n; ++i) {
        scyte_node* node = nodes[i];
        printf("%d\t", i);
        putchar('[');
        for(j = 0; j < node->num_dims; ++j) {
            if(j) putchar(',');
            printf("%d", node->shape[j]);
        }
        printf("]\t\t");
        if(node->num_children > 0) {
            printf("%s(", scyte_get_op_string(node->op_type));
            for(j = 0; j < node->num_children; ++j) {
                if(j) putchar(',');
                printf("$%d", node->children[j]->mark);
            }
            printf(")");
        }
        else printf("%s", scyte_is_var(node) ? "var" : scyte_is_const(node) ? "const"
                        : scyte_is_input(node) ? "input" : "N/A");
        putchar('\n');
    }
    printf("----------------------------\n");
    for(i = 0; i < n; ++i) nodes[i]->mark = 0;
}
