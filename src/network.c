#define SCYTE_VERBOSE
#include "network.h"

#include "logger.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <float.h>

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

static inline int get_placeholder_dim(scyte_network* net, scyte_node_type type)
{
    int count = 0, dim = -1;
    for(int i = 0; i < net->n; ++i) {
        scyte_node* node = net->nodes[i];
        if(scyte_is_placeholder(node) && (node->type & type)) {
            ++count;
            if(node->num_dims > 1) dim = scyte_num_elements(node) / node->shape[0]; // divide by batch size
            else if(node->num_dims == 1) dim = node->shape[0]; // vector
            else dim = 1; // scalar
        }
    }
    return count == 1 ? dim : -1;
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

static inline float scyte_calculate_cost(scyte_network* net, int calc_grads)
{
    int cost_idx = scyte_find_node(net, COST);
    if(cost_idx < 0) {
        LOG_ERROR("couldn't find any cost node");
        assert(0);
    }
    float cost = *scyte_forward(net->n, net->nodes, cost_idx);
    if(calc_grads) scyte_backward(net->n, net->nodes, cost_idx);
    return cost;
}

static inline void optimizer_step(scyte_optimizer_params params, scyte_network* net, int n, float* g_prev, float* g_mean, float* g_var)
{
    if(params.type == ADAM) scyte_adam_step(params, n, net->deltas, g_var, g_mean, net->vals);
    else if(params.type == RMSPROP) scyte_rmsprop_step(params, n, net->deltas, g_var, net->vals);
    else if(params.type == SGD) scyte_sgd_step(params, n, net->deltas, g_prev, net->vals);
}

void scyte_train_network(scyte_network* net, scyte_optimizer_params params, int batch_size, int num_epochs, float val_split, int early_stop_patience, scyte_data data)
{
    int n = data.X.rows;
    int num_in = get_placeholder_dim(net, INPUT), num_target = get_placeholder_dim(net, GROUND_TRUTH);
    assert(num_in == data.X.cols && num_target == data.y.cols);
    int num_vars = get_num_vars(net), num_consts = get_num_consts(net);
    // temporary buffers used for storing the best network values, based on validation metrics
    float* best_vals = (float*)malloc(num_vars*sizeof(float));
    float* best_consts = (float*)malloc(num_consts*sizeof(float));

    float* g_var = NULL, *g_mean = NULL, *g_prev = NULL;
    if(params.type == SGD) g_prev = (float*)calloc(num_vars, sizeof(float));
    else if(params.type == RMSPROP || params.type == ADAM) {
        g_var = (float*)calloc(num_vars, sizeof(float));
        if(params.type == ADAM) g_mean = (float*)calloc(num_vars, sizeof(float));
    }

    float* X = (float*)malloc(num_in*batch_size*sizeof(float));
    float* y = (float*)malloc(num_target*batch_size*sizeof(float));
    scyte_feed_net(net, INPUT, &X); // input node will be binded to input array
    scyte_feed_net(net, GROUND_TRUTH, &y); // ground truth node will be binded to target array

    int num_val = n*val_split, num_train = n - num_val, keep_best = 0, no_improvement_count = 0;
    float best_val_cost = FLT_MAX;
    for(int i = 0; i < num_epochs; ++i) {
        double t1 = time_now();
        int num_processed = 0;
        float train_cost = 0.f, val_cost = 0.f;
        // training
        switch_propagation_mode(net, 1);
        while(num_processed < num_train) {
            int bs = num_train - num_processed < batch_size ? num_train - num_processed : batch_size;
            scyte_random_batch(data, bs, X, y);
            train_cost += bs*scyte_calculate_cost(net, 1);
            optimizer_step(params, net, num_vars, g_prev, g_mean, g_var);
            num_processed += bs;
        }
        train_cost /= num_train;

        // validation
        num_processed = 0;
        switch_propagation_mode(net, 0);
        while(num_processed < num_val) {
            int bs = num_val - num_processed < batch_size ? num_val - num_processed : batch_size;
            scyte_random_batch(data, bs, X, y);
            val_cost += bs*scyte_calculate_cost(net, 0);
            num_processed += bs;
        }
#ifdef SCYTE_VERBOSE
        fprintf(stderr, "epoch %d – training cost: %.3f ", i+1, train_cost);
        if(num_val > 0) {
            val_cost /= num_val;
            fprintf(stderr, "- validation cost: %g ", val_cost);
        }
        double t2 = time_now();
        fprintf(stderr, "- %.3gs/iter", t2 - t1);
        fprintf(stderr, "\n");
#endif
        // early stopping if no changes for early_stop_patience epochs
        if(num_val > 0 && i >= early_stop_patience) {
            if(val_cost < best_val_cost) {
                keep_best = 1, no_improvement_count = 0, best_val_cost = val_cost;
                memcpy(best_vals, net->vals, num_vars*sizeof(float));
                memcpy(best_consts, net->consts, num_consts*sizeof(float));
            }
            else if(++no_improvement_count >= early_stop_patience) break;
        }
    }
    if(num_val > 0 && keep_best) {
        memcpy(net->vals, best_vals, num_vars*sizeof(float));
        memcpy(net->consts, best_consts, num_consts*sizeof(float));
    }
    free(best_vals); free(best_consts); free(X); free(y);
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
