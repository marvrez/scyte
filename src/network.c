#include "network.h"

#include "logger.h"

#include <stdlib.h>

// switch between forward and backward propagation mode
static inline void switch_propagation_mode(scyte_network* net, int is_backward)
{
    for(int i = 0; i < net->n; ++i) {
        scyte_node* node = net->nodes[i];
        if(node->op_type == SELECT && node->num_children == 2) {
            *(int*)node->params = !!is_backward;
        }
    }
}

// get number of variables in the network
static inline int get_num_vars(scyte_network* net)
{
    int count = 0;
    for(int i = 0; i < net->n; ++i) {
        scyte_node* node = net->nodes[i];
        if(scyte_is_var(node)) {
            count += scyte_num_elements(node);
        }
    }
    return count;
}

// get number of constants in the network
static inline int get_num_consts(scyte_network* net)
{
    int count = 0;
    for(int i = 0; i < net->n; ++i) {
        scyte_node* node = net->nodes[i];
        if(scyte_is_const(node)) {
            count += scyte_num_elements(node);
        }
    }
    return count;
}

static inline void alloc_network(scyte_network* net)
{
    int j = 0, k = 0;
    int num_vars = get_num_vars(net), num_consts = get_num_consts(net);
    net->vals = (float*)realloc(net->vals, num_vars*sizeof(float));
    net->deltas = (float*)realloc(net->deltas, num_vars*sizeof(float));
    net->consts = (float*)realloc(net->consts, num_consts*sizeof(float));
    memset(net->deltas, 0, num_vars*sizeof(float));
    for(int i = 0; i < net->n; ++i) {
        scyte_node* node = net->nodes[i];
        int num_elements = scyte_num_elements(node);
        if(scyte_is_var(node)) {
            memcpy(&net->vals[j], node->vals, num_elements*sizeof(float));
            free(node->vals);
            node->vals = &net->vals[j];
            node->delta = &net->deltas[j];
            j += num_elements;
        }
        else if(scyte_is_const(node)) {
            memcpy(&net->consts[k], node->vals, num_elements*sizeof(float));
            free(node->vals);
            node->vals = &net->consts[k];
            k += num_elements;
        }
    }
}

scyte_network* scyte_make_network(scyte_node* cost_node)
{
    return scyte_make_network2(cost_node, 0, NULL);
}

scyte_network* scyte_make_network2(scyte_node* cost_node, int num_other_roots, scyte_node** other_roots)
{
    if(cost_node->num_dims != 0) {
        LOG_ERROR("couldn't make network, cost node must output a scalar");
        return NULL;
    }
    int num_roots = 1 + num_other_roots, i;
    scyte_network* net = (scyte_network*)calloc(1, sizeof(scyte_network));
    scyte_node** roots = (scyte_node**)malloc(num_roots*sizeof(scyte_node*));
    for(i = 0; i < num_other_roots; ++i) roots[i] = other_roots[i];
    roots[i] = cost_node;
    net->nodes = scyte_make_graph(&net->n, num_roots, roots);
    alloc_network(net);
    free(roots);
    return net;
}

const float* scyte_predict_network(scyte_network* net, float* data)
{
    int out_idx = scyte_find_node(net, OUTPUT);
    if(out_idx < 0) {
        LOG_ERROR("couldn't find any output node");
        return NULL;
    }
    scyte_feed_net(net, INPUT, &data);
    return scyte_forward(net->n, net->nodes, out_idx);
}

void scyte_free_network(scyte_network* net)
{
    if(!net) return;
    free(net->vals); free(net->deltas); free(net->consts);
    scyte_free_graph(net->n, net->nodes);
    free(net);
}

void scyte_save_network(const char* filename, scyte_network* net)
{
    FILE* fp = fopen(filename, "wb");
    fwrite("SCYTE", sizeof(char), 5, fp); // magic number memes
    scyte_save_graph(fp, net->n, net->nodes);
    fwrite(net->vals, sizeof(float), get_num_vars(net), fp);
    fwrite(net->consts, sizeof(float), get_num_consts(net), fp);
    fclose(fp);
}

// synchronizes nodes in a network with global variables such as consts and variables
static inline void sync_network(scyte_network* net)
{
    int j = 0, k = 0;
    for(int i = 0; i < net->n; ++i) {
        scyte_node* node = net->nodes[i];
        int num_elements = scyte_num_elements(node);
        if(scyte_is_var(node)) {
            node->vals = &net->vals[j];
            node->delta = &net->deltas[j];
            j += num_elements;
        }
        else if(scyte_is_const(node)) {
            node->vals = &net->consts[k];
            k += num_elements;
        }
    }
}

scyte_network* scyte_load_network(const char* filename)
{
    FILE* fp = fopen(filename, "rb");
    // parse and verify magic number
    char magic_str[5];
    fread(magic_str, sizeof(char), 5, fp);
    if(strncmp(magic_str, "SCYTE", 5) != 0) {
        LOG_ERROR("couldn't load file: magic number didn't match");
        fclose(fp);
        return NULL;
    }
    scyte_network* net = (scyte_network*)calloc(1, sizeof(scyte_network));
    net->nodes = scyte_load_graph(fp, &net->n);
    int num_vars = get_num_vars(net), num_consts = get_num_consts(net);
    net->vals = (float*)malloc(num_vars*sizeof(float));
    net->deltas = (float*)malloc(num_vars*sizeof(float));
    net->consts = (float*)malloc(num_consts*sizeof(float));
    fread(net->vals, sizeof(float), num_vars, fp);
    fread(net->consts, sizeof(float), num_consts, fp);
    sync_network(net);
    fclose(fp);
    return net;
}
