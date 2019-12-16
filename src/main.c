#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "scyte.h"
#include "op.h"
#include "layers.h"
#include "network.h"
#include "optimizer.h"
#include "data.h"

#include "blas.h"
#include "logger.h"
#include "utils.h"

int main(int argc, char** argv)
{
    srand(time(NULL));
    scyte_data d;
    float data[4][2] = {{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}};
    d.X.data = (float*)data, d.X.rows = 4, d.X.cols = 2;
    float target[4][1] = {{0.f}, {1.f}, {1.f}, {0.f}};
    d.y.data = (float*)target, d.y.rows = 4, d.y.cols = 1;

    float x_test[2] = { 0.f, 1.f };
#if 1
    double t1 = time_now();
    scyte_node* in = scyte_layer_input(2);
    scyte_node* dense  = scyte_sigmoid(scyte_layer_connected(in, 4));
    scyte_node* loss = scyte_layer_cost(dense, 1, COST_BINARY_CROSS_ENTROPY);

    scyte_network* net = scyte_make_network(loss);
    int num = net->n;

    scyte_print_graph(num, net->nodes);

    scyte_optimizer_params params = scyte_sgd_params(0.01f, 0.0005f, 0.90f);
    scyte_train_network(net, params, 1, 5000, 0, 20, d);

    double t2 = time_now();
    LOG_INFOF("%.3lf seconds", t2-t1);
    scyte_save_network("test.weight", net);
    LOG_INFO("network saved");
#else
    scyte_network* net = scyte_load_network("test.weight");
    scyte_print_graph(net->n, net->nodes);
#endif

    x_test[0]=0.f, x_test[1] = 0.f;
    const float* vals_new = scyte_predict_network(net, x_test);
    printf("%f %f --> %f\n", x_test[0], x_test[1], *vals_new);

    x_test[0]=0.f, x_test[1] = 1.f;
    vals_new = scyte_predict_network(net, x_test);
    printf("%f %f --> %f\n", x_test[0], x_test[1], *vals_new);

    x_test[0]=1.f, x_test[1] = 0.f;
    vals_new = scyte_predict_network(net, x_test);
    printf("%f %f --> %f\n", x_test[0], x_test[1], *vals_new);

    x_test[0]=1.f, x_test[1] = 1.f;
    vals_new = scyte_predict_network(net, x_test);
    printf("%f %f --> %f\n", x_test[0], x_test[1], *vals_new);
    scyte_free_network(net);
    return 0;
}
