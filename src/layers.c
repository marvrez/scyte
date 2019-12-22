#include "layers.h"

#include "op.h"
#include "logger.h"
#include "utils.h"

#include <stdio.h>

const char* get_cost_string(cost_type type)
{
    switch(type) {
        case COST_BINARY_CROSS_ENTROPY: return "binary_cross_entropy";
        case COST_CROSS_ENTROPY: return "cross_entropy";
        case COST_L1: return "L1";
        case COST_L2: return "L2";
    }
    return "unknown";
}

scyte_node* scyte_layer_input(int n)
{
    int shape[] = {1, n};
    fprintf(stderr, "input                            %4d\n", n);
    scyte_node* node = scyte_placeholder(2, shape);
    node->type |= INPUT;
    return node;
}

scyte_node* scyte_layer_input_image(int w, int h, int c)
{
    int shape[] = {1, c, h, w};
    fprintf(stderr, "input                            %4d x %4d x %4d\n", w, h, c);
    scyte_node* node = scyte_placeholder(4, shape);
    node->type |= INPUT;
    return node;
}

scyte_node* scyte_layer_connected(scyte_node* in, int num_outputs)
{
    // divide by by batch_size
    int num_inputs = in->num_dims >= 2 ? 
        scyte_num_elements(in) / in->shape[0] : scyte_num_elements(in);
    fprintf(stderr, "connected                        %4d -> %4d     %5.3g BFLOPS\n", num_inputs, num_outputs, 2.f*num_inputs*num_outputs/1e9f);
    scyte_node* w = scyte_weight(num_outputs, num_inputs);
    scyte_node* b = scyte_bias(num_outputs, 0.0f);
    return scyte_add(scyte_cmatmul(in, w), b);
}

scyte_node* scyte_layer_dropout(scyte_node* in, float dropout_rate)
{
    scyte_node* rate = scyte_scalar(CONST, dropout_rate);
    scyte_node* switch_nodes[] = { in, scyte_dropout(in, rate) };
    fprintf(stderr, "dropout        p=%.2f            %4d\n", dropout_rate, scyte_num_elements(in));
    return scyte_dynamic_select(2, switch_nodes); 
}

scyte_node* scyte_layer_layernorm(scyte_node* in)
{

    int dim = in->num_dims >= 2 ?
            scyte_num_elements(in) / in->shape[0] : scyte_num_elements(in);
    char* shape_str = get_shape_string(in->num_dims, in->shape);
    fprintf(stderr, "layer_norm                          %s\n", shape_str);

    scyte_node* alpha = scyte_bias(dim, 1.f);
    scyte_node* beta  = scyte_bias(dim, 0.f);

    free(shape_str);

    return scyte_add(scyte_mul(scyte_normalize(in), alpha), beta);
}

scyte_node* scyte_layer_cost(scyte_node* in, int num_out, cost_type type)
{
    scyte_node* pred = scyte_layer_connected(in, num_out);

    int out_shape[] = {1, num_out};
    scyte_node* truth = scyte_placeholder(2, out_shape);

    scyte_node* cost = NULL;
    switch(type) {
        case COST_BINARY_CROSS_ENTROPY:
            pred = scyte_sigmoid(pred);
            cost = scyte_logistic_x_entropy(truth, pred);
            break;
        case COST_CROSS_ENTROPY:
            pred = scyte_softmax(pred);
            cost = scyte_categorical_x_entropy(truth, pred);
            break;
        case COST_L1:
            cost = scyte_l1_norm(truth, pred);
            break;
        case COST_L2:
            cost = scyte_mse(truth, pred);
            break;
        default:
            LOG_ERROR("Unknown cost type");
            break;
    }
    pred->type |= OUTPUT, cost->type |= COST, truth->type |= GROUND_TRUTH;
    fprintf(stderr, "cost                             %4d (%s)\n", num_out, get_cost_string(type));
    return cost;
}


scyte_node* scyte_layer_maxpool2d(scyte_node* in, int size, int stride, int padding)
{
    scyte_node* out = scyte_maxpool2d(in, size, stride, padding);
    int c = in->shape[1], h = in->shape[2], w = in->shape[3];
    int out_c = out->shape[1], out_h = out->shape[2], out_w = out->shape[3];
    fprintf(stderr, "maxpool2d                        %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, out_w, out_h, out_c);
    return out;
}

scyte_node* scyte_layer_conv2d(scyte_node* in, int num_filters, int size, int stride, int padding)
{
    int c = in->shape[1], h = in->shape[2], w = in->shape[3];
    int filter_shape[4] = { num_filters, c, size, size };
    scyte_node* filter_weight = scyte_var(4, filter_shape, 0.f);
    scyte_node* out = scyte_conv2d(in, filter_weight, stride, padding);

    int out_c = out->shape[1], out_h = out->shape[2], out_w = out->shape[3];
    fprintf(stderr, "conv  %5d %2d x%2d / %2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3g BFLOPs\n", num_filters, size, size, stride, w, h, c, out_w, out_h, out_c, (2.f*num_filters*size*size*c*out_h*out_w)/1e9f);

    return out;
}
