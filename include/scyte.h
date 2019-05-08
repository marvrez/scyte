#ifndef SCYTE_H
#define SCYTE_H

#include <stdio.h>

#define SCYTE_MAX_DIMS 4

typedef enum {
    INPUT = 0x0,
    VAR = 0x1,
    CONST = 0x2,
} scyte_node_type;

#define scyte_has_gradient(p)  ((p)->type & VAR)

#define scyte_is_operand(p)     ((p)->num_children == 0)
#define scyte_is_var(p)         (scyte_is_operand(p) && scyte_has_gradient(p))
#define scyte_is_const(p)       (scyte_is_operand(p) && ((p)->type & CONST))
#define scyte_is_input(p)       (scyte_is_operand(p) && !scyte_has_gradient(p) && !((p)->type & CONST))

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

    float*      vals;   // stored values for node
    float*      delta;  // deltas provided by the output/top nodes

    float*      tmp;    // values produced in forward pass that are needed for the backward pass
    float*      params; // extra parameters needed by the node, e.g. stride and padding for convolution

    int         mark;   // describes the temporary state of a node, used internally

    int                 num_children;
    struct scyte_node** children;
} scyte_node;

typedef struct {
    int n;              // number of nodes in the network
    scyte_node** nodes; // array of the nodes in the network
} scyte_network;

// node->vals are set to fill_val if num_dims <=  1
scyte_node* scyte_input(unsigned num_dims, int shape[SCYTE_MAX_DIMS]);
scyte_node* scyte_const(unsigned num_dims, int shape[SCYTE_MAX_DIMS], float fill_val);
scyte_node* scyte_var(unsigned num_dims, int shape[SCYTE_MAX_DIMS], float fill_val);

scyte_node** scyte_make_graph(int* num_nodes, int num_roots, scyte_node** roots);
void scyte_free_graph(int n, scyte_node** nodes);
scyte_node** scyte_copy_graph(int n, scyte_node** nodes, int batch_size);

void scyte_copy_dim(const scyte_node* src, scyte_node* dst);

// returns a pointer to nodes[to]->vals, so be careful to not free the data!
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
