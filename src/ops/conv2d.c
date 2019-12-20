#include "ops/conv2d.h"

#include "op.h"
#include "logger.h"


static inline int convolutional_out_width(scyte_node* node)
{

}

static inline int convolutional_out_height(scyte_node* node)
{

}

int scyte_conv2d_sync_dims(scyte_node* node)
{

}

static inline void set_conv_params(scyte_node* node, int stride, int padding)
{

}

scyte_node* scyte_conv2d(scyte_node* x, scyte_node* w, int stride, int padding)
{
    if(x->num_dims != 4 || w->num_dims != 4) {
        LOG_ERROR("input or weights must have dim 4");
        return NULL;
    }
    scyte_node* node = make_op2_node(CONV2D, x, w);
    // set-up output shape and save parameters
    set_conv_params(node, stride, padding);
    if(!scyte_conv2d_sync_dims(node)) {
        free_op_node(node);
        return NULL;
    }
    return node;
}

void scyte_conv2d_forward(scyte_node* node)
{

}

void scyte_conv2d_backward(scyte_node* node)
{

}
