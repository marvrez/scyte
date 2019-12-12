#ifndef SCYTE_H
#define SCYTE_H

#include <stdio.h>

#define SCYTE_MAX_DIMS 4

typedef unsigned char scyte_node_type;

#define PLACEHOLDER     0x1
#define VAR             0x2
#define CONST           0x4
#define INPUT           0x8
#define OUTPUT          0x10
#define GROUND_TRUTH    0x20
#define COST            0x40

#define scyte_has_gradient(p)   ((p)->type & VAR)

#define scyte_is_operand(p)     ((p)->num_children == 0)
#define scyte_is_var(p)         (scyte_is_operand(p) && scyte_has_gradient(p))
#define scyte_is_const(p)       (scyte_is_operand(p) && ((p)->type & CONST))
#define scyte_is_placeholder(p) (scyte_is_operand(p) && !scyte_has_gradient(p) && !((p)->type & CONST))

typedef enum {
    NOOP = 0,
    ADD,
    SUB,
    MULTIPLY,
    SQUARE,
    SIGMOID,
    TANH,
    RELU,
    MATMUL,
    CMATMUL,
    AVG,
    MAX,
    SELECT,
    DROPOUT,
    SOFTMAX,
    EXP,
    LOG,
    SIN,
    MSE,
    L1_NORM,
    LOGXENT,
    CATEGORICALXENT,
    REDUCE_MEAN,
    REDUCE_SUM,
    RESHAPE,
    CONCAT,
    SLICE,
    NORMALIZE,
    NOP,
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

    void*       tmp;    // values produced in forward pass that are needed for the backward pass
    void*       params; // extra parameters needed by the node, e.g. stride and padding for convolution
    size_t      params_size;

    int         mark;   // describes the temporary state of a node, used internally

    int                 num_children;
    struct scyte_node** children;
} scyte_node;

typedef struct {
    int n;              // number of nodes in the network
    scyte_node** nodes; // array of the nodes in the network
    float* vals;    // collated values
    float* deltas;  // collated deltas
    float* consts;  // collated constants
} scyte_network;

// node->vals are set to fill_val if num_dims <= 1
scyte_node* scyte_placeholder(unsigned num_dims, int shape[SCYTE_MAX_DIMS]);
scyte_node* scyte_scalar(scyte_node_type type, float val);
scyte_node* scyte_bias(int n, float default_val);
scyte_node* scyte_weight(int rows, int cols);


// find index of node with certain type
int scyte_find_node(scyte_network* net, scyte_node_type type);
// feed placeholders in a net of a certain type with given vals
int scyte_feed_net(scyte_network* net, scyte_node_type type, float** vals);
// feed a single placeholder
void scyte_feed_placeholder(scyte_node* node, float* vals);

scyte_node** scyte_make_graph(int* num_nodes, int num_roots, scyte_node** roots);
void scyte_free_graph(int n, scyte_node** nodes);
scyte_node** scyte_copy_graph(int n, scyte_node** nodes, int batch_size);

void scyte_copy_shape(const scyte_node* src, scyte_node* dst);
void scyte_fill_vals(scyte_node* node, float fill_val);

// returns a pointer to nodes[to]->vals
const float* scyte_forward(int n, scyte_node** nodes, int to);
void scyte_backward(int n, scyte_node** nodes, int from);

void scyte_print_graph(int n, scyte_node** nodes);
void scyte_save_graph(FILE* fp, int num_nodes, scyte_node** nodes);
scyte_node** scyte_load_graph(FILE* fp, int* n);

static inline int scyte_num_elements(scyte_node* node)
{
    int n = 1;
    for(int i = 0; i < node->num_dims; ++i) {
        n *= node->shape[i];
    }
    return n;
}

#endif
