#ifndef SCYTE_H
#define SCYTE_H

#include <stdio.h>

#define SCYTE_MAX_DIMS 4

typedef enum {
    VAR = 0x1,
    CONST = 0x2,
} scyte_node_type;

typedef enum {
    ADD,
    SUB,
    MULTIPLY,
    SQUARE,
    SIGMOID,
    TANH,
    RELU,
    MATMUL,
    AVG,
    DROPOUT,
    MAX,
    SOFTMAX,
    EXP,
    UNKNOWN,
} scyte_op_type;

typedef struct scyte_node {
    scyte_node_type type; // type of node – var, const, etc.
    scyte_op_type op_type; // type of operation – add, multiply, subtract, etc.

    void    (*forward)  (struct scyte_node*);
    void    (*backward) (struct scyte_node*);

    unsigned    num_dims;
    int         shape[SCYTE_MAX_DIMS];

    float*      in;     // input values to the node
    float*      out;    // output values produced by the node
    float*      delta;  // deltas provided by the output/top nodes

    float*      tmp;    // values produced in forward pass that are needed for the backward pass
    float*      params; // extra parameters needed by the node, e.g. stride and padding for convolution

    int         mark;   // describes the temporary state of a node, used internally

    int                 num_children;
    struct scyte_node** children;
} scyte_node;

scyte_node** scyte_make_graph(int* num_nodes, int num_roots, scyte_node** roots);
void scyte_free_graph(int n, scyte_node** nodes);
scyte_node** scyte_copy_graph(int n, scyte_node** nodes, int batch_size);

// returns a pointer to nodes[to]->out, so be careful to not free the data!
const float* scyte_forward(int n, scyte_node** nodes, int to);
void scyte_backward(int n, scyte_node** nodes, int from);

void scyte_print_graph(int n, scyte_node** nodes);
int scyte_save_graph(FILE* fp, int num_nodes, scyte_node** nodes);
scyte_node** scyte_load_graph(FILE* fp, int* num_nodes);

static inline int scyte_num_elements(scyte_node* node)
{
    int n = 1;
    for(int i = 0; i < node->num_dims; ++i) {
        n *= node->shape[i];
    }
    return n;
}

#endif
