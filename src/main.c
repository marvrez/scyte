#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "scyte.h"
#include "op.h"
#include "layers.h"
#include "network.h"

#include "blas.h"
#include "logger.h"
#include "utils.h"

int main(int argc, char** argv)
{
    srand(time(NULL));
    float mems[2] = {1};
#if 1
    double t1 = time_now();
    scyte_node* in = scyte_layer_input(2);
    scyte_node* dense  = scyte_sigmoid(scyte_layer_connected(in, 8));
    scyte_node* dense2  = scyte_sigmoid(scyte_layer_connected(dense, 16));
    scyte_node* loss = scyte_layer_cost(dense2, 1, COST_L2);

    scyte_network* net = scyte_make_network(loss);
    int num = net->n;

    scyte_print_graph(num, net->nodes);
    const float* vals = scyte_predict_network(net, mems);
    double t2 = time_now();
    printf("output vals %f\n", vals[0]);
    printf("after %f\n", dense2->vals[0]);
    LOG_INFOF("%.3lf seconds", t2-t1);
    scyte_save_network("test.weight", net);
    LOG_INFO("network saved");
#else
    scyte_network* net = scyte_load_network("test.weight");
    scyte_print_graph(net->n, net->nodes);
    printf("%f\n", net->vals[0]);
    const float* vals = scyte_predict_network(net, mems);
    printf("output vals %f\n", vals[0]);
#endif
    scyte_free_network(net);

    return 0;
}
