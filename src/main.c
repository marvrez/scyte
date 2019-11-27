#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "scyte.h"
#include "op.h"
#include "layers.h"

#include "blas.h"
#include "logger.h"
#include "utils.h"

int main(int argc, char** argv)
{
    srand(time(NULL));
#if 1
    scyte_node* a = scyte_layer_input(3);
    scyte_node* dense  = scyte_tanh(scyte_relu(scyte_sigmoid(scyte_layer_connected(a, 9))));
    int shape[] = {3, 3};
    scyte_node* dense2 = scyte_reshape(dense, 2, shape);
    scyte_layer_dropout(dense2, 0.7);
    scyte_layer_layernorm(dense2);
    scyte_slice(dense2, 0, 0, 1);
    scyte_node* test[] = {dense, dense};
    scyte_concat(0, 2, test);
    scyte_reduce_mean(dense2, 0);
    scyte_node* meme = scyte_select(0,2, test);
    scyte_node* loss = scyte_layer_cost(dense, 9, COST_CROSS_ENTROPY);

    scyte_node* nodes[] = {loss};

    int num;
    scyte_node** graph = scyte_make_graph(&num, sizeof(nodes) / sizeof(scyte_node*), nodes);
    //scyte_print_graph(num, graph);
    double t1 = time_now();

    float mems[3] = {1};
    scyte_feed_placeholder(a, mems);

    printf("before %f\n", dense->vals[0]);
    const float* vals = scyte_forward(num, graph, 5);
    double t2 = time_now();
    printf("after %f\n", dense->vals[0]);
    printf("after vals %f\n", vals[0]);
    LOG_INFOF("%.3lf seconds", t2-t1);
    //scyte_backward(num, graph, -1);
    FILE* fp = fopen("test.weight", "w");
    scyte_save_graph(fp, num, graph);
#else
    int num;
    FILE* fp = fopen("test.weight", "r");
    scyte_node** graph = scyte_load_graph(fp, &num);
    scyte_print_graph(num, graph);
#endif
    scyte_free_graph(num, graph);

    return 0;
}
