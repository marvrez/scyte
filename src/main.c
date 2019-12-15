#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "scyte.h"
#include "op.h"
#include "layers.h"
#include "network.h"
#include "optimizer.h"

#include "blas.h"
#include "logger.h"
#include "utils.h"

int main(int argc, char** argv)
{
    srand(time(NULL));
    float data[4][2] = {{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}};
    float* data2[4] = { data[0], data[1], data[2], data[3]};
    float** x = data2;

    float target[4][1] = {{0.f}, {1.f}, {1.f}, {0.f}};
    float* target2[4] = { target[0], target[1], target[2], target[3]};
    float** y = target2;

    float x_test[2] = {0.f, 1.f};
#if 1
    double t1 = time_now();
    scyte_node* in = scyte_layer_input(2);
    scyte_node* dense  = scyte_relu(scyte_layer_connected(in, 8));
    scyte_node* dense2  = scyte_relu(scyte_layer_connected(dense, 16));
    scyte_node* loss = scyte_layer_cost(dense2, 1, COST_BINARY_CROSS_ENTROPY);

    scyte_network* net = scyte_make_network(loss);
    int num = net->n;

    scyte_print_graph(num, net->nodes);
    const float* vals = scyte_predict_network(net, x_test);
    printf("before train: %f\n", vals[0]);

    scyte_optimizer_params params = scyte_rmsprop_params(0.01, 0.f, 0.f, 0.99);
    scyte_train_network(net, params, 1, 500, 0, 20, 4, x, y);
    const float* vals_new = scyte_predict_network(net, x_test);
    printf("after train: %f\n", vals_new[0]);


    double t2 = time_now();
    LOG_INFOF("%.3lf seconds", t2-t1);
    scyte_save_network("test.weight", net);
    LOG_INFO("network saved");
#else
    scyte_network* net = scyte_load_network("test.weight");
    scyte_print_graph(net->n, net->nodes);
    const float* vals = scyte_predict_network(net, x);
    printf("output vals %f\n", vals[0]);
#endif
    scyte_free_network(net);
    return 0;
}
