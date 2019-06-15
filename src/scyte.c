#include "scyte.h"
#include "op.h"

#include "blas.h"
#include "list.h"
#include "logger.h"
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
        if(node->num_dims <= 1) set_cpu(num_elements, fill_val, node->vals);
        else {
            float s = 2.f / sqrtf((float)num_elements / node->shape[0]); // s = 2 / sqrt(n_in)
            #pragma omp parallel for
            for(int i = 0; i < num_elements; ++i) {
                node->vals[i] = s*random_uniform(-1, 1);
            }
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

static inline void scyte_propagate_gradient_marks(int n, scyte_node** nodes)
{
    for(int i = 0; i < n; ++i) {
        int j;
        scyte_node* node = nodes[i];
        if(node->num_children == 0) continue;
        for(j = 0; j < node->num_children; ++j) {
            if(scyte_has_gradient(node->children[j])) {
                break;
            }
        }
        if(j < node->num_children) node->type |= VAR;
        else node->type &= ~VAR;
    }
}

static void scyte_allocate_tensors(int n, scyte_node** nodes)
{
    scyte_propagate_gradient_marks(n, nodes);
    for(int i = 0; i < n; ++i) {
        scyte_node* node = nodes[i];
        if(node->num_children == 0) continue;
        int num_elements = scyte_num_elements(node);
        node->vals = (float*)realloc(node->vals, num_elements*sizeof(float));
        if(scyte_has_gradient(node)) {
            node->delta = (float*)realloc(node->delta, num_elements*sizeof(float));
        }
    }
}

scyte_node** scyte_make_graph(int* num_nodes, int num_roots, scyte_node** roots)
{
    list* l = make_list();
    list* out = make_list();
    // `mark` is the in-degree count, left shifted by 1
    for(int i = 0; i < num_roots; ++i) {
        roots[i]->mark = 1;
        list_append(l, roots[i]);
    }

    // traverse graph to calculate in-degrees of the nodes
    while(l->size) {
        scyte_node* node = list_pop(l);
        for(int i = 0; i < node->num_children; ++i) {
            scyte_node* child = node->children[i];
            // child hasn't been visited – explore it
            if(child->mark == 0) list_append(l, child);
            child->mark += 0x2;
        }
    }
    // push all nodes with in-degree 0
    for(int i = 0; i < num_roots; ++i) {
        if(roots[i]->mark >> 1 == 0) {
            list_append(l, roots[i]);
        }
    }
    // perform kahn's algorithm to topological sort the graph
    while(l->size) {
        scyte_node* node = list_pop(l);
        list_append(out, node);
        // iterate through all childrens and decrease their in-degree by 2
        // (since it's left-shifted by 1)
        for(int i = 0; i < node->num_children; ++i) {
            scyte_node* child = node->children[i];
            child->mark -= 0x2;
            if(child->mark >> 1 == 0) list_append(l, child);
        }
    }

    // check for cycles
    list_node* n = out->head;
    while(n) {
        scyte_node* node = n->data;
        if(node->mark >> 1 != 0) {
            LOG_ERROR("detected a cycle in the computational graph");
            assert(0);
        }
        node->mark = 0x0;
        n = n->next;
    }
    scyte_node** graph = (scyte_node**)list_to_reverse_array(out);
    *num_nodes = out->size;
    scyte_allocate_tensors(*num_nodes, graph);

    free_list(l);
    return graph;
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
        if(node->num_children > 0 && node->mark > 0) {
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

static inline void scyte_save_node(FILE* fp, scyte_node* node)
{
    fwrite(&node->type, sizeof(scyte_node_type), 1, fp);
    fwrite(&node->num_children, sizeof(int), 1, fp);
    if(node->num_children > 0) {
        fwrite(&node->op_type, sizeof(scyte_op_type), 1, fp);
        for(int j = 0; j < node->num_children; ++j) {
            fwrite(&node->children[j]->mark, sizeof(int), 1, fp);
        }
        fwrite(&node->params_size, sizeof(size_t), 1, fp);
        if(node->params_size > 0 && node->params) {
            fwrite(node->params, node->params_size, 1, fp);
        }
    }
    fwrite(&node->num_dims, sizeof(unsigned), 1, fp);
    if(node->num_dims > 0) fwrite(node->shape, sizeof(int), node->num_dims, fp);
}

void scyte_save_graph(FILE* fp, int num_nodes, scyte_node** nodes)
{
    fwrite(&num_nodes, sizeof(int), 1, fp);
    for(int i = 0; i < num_nodes; ++i) nodes[i]->mark = i;
    for(int i = 0; i < num_nodes; ++i) scyte_save_node(fp, nodes[i]);
    for(int i = 0; i < num_nodes; ++i) nodes[i]->mark = 0;
}

static inline scyte_node* scyte_load_node(FILE* fp, scyte_node** graph)
{
    scyte_node* node = (scyte_node*)calloc(1, sizeof(scyte_node));
    fread(&node->type, sizeof(scyte_node_type), 1, fp);
    fread(&node->num_children, sizeof(int), 1, fp);
    if(node->num_children > 0) {
        int child_idx;
        node->children = (scyte_node**)calloc(node->num_children, sizeof(scyte_node*));
        fread(&node->op_type, sizeof(scyte_op_type), 1, fp);
        for(int j = 0; j < node->num_children; ++j) {
            fread(&child_idx, sizeof(int), 1, fp);
            node->children[j] = graph != NULL ? graph[child_idx] : 0;
        }
        fread(&node->params_size, sizeof(size_t), 1, fp);
        if(node->params_size > 0) {
            node->params = malloc(node->params_size);
            fread(node->params, node->params_size, 1, fp);
        }
    }
    fread(&node->num_dims, sizeof(unsigned), 1, fp);
    if(node->num_dims > 0) fread(node->shape, sizeof(int), node->num_dims, fp);
    return node;
}

scyte_node** scyte_load_graph(FILE* fp, int* n)
{
    int num_nodes;
    scyte_node** graph;

    fread(&num_nodes, sizeof(int), 1, fp);
    graph = (scyte_node**)calloc(num_nodes, sizeof(scyte_node*));
    for(int i = 0; i < num_nodes; ++i) {
        graph[i] = scyte_load_node(fp, graph);
    }
    *n = num_nodes;
    scyte_propagate_gradient_marks(num_nodes, graph);
    return graph;
}
