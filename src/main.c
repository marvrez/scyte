#include <stdio.h>

#include "scyte.h"
#include "network.h"
#include "data.h"

#include "logger.h"
#include "utils.h"

scyte_network* make_model()
{
    scyte_node* in = scyte_layer_input(2);
    scyte_node* dense  = scyte_sigmoid(scyte_layer_connected(in, 4));
    scyte_node* loss = scyte_layer_cost(dense, 1, COST_BINARY_CROSS_ENTROPY);
    return scyte_make_network(loss);
}

static inline scyte_data generate_xor_data()
{
    scyte_data d;
    float** X = malloc(4*sizeof(float*)), **y = malloc(4*sizeof(float*));
    static float data[4][2] = {{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}};
    static float target[4][1] = {{0.f}, {1.f}, {1.f}, {0.f}};
    for(int i = 0; i < 4; ++i) X[i] = data[i];
    d.X.data = X; d.X.rows = 4, d.X.cols = 2;
    for(int i = 0; i < 4; ++i) y[i] = target[i];
    d.y.data = y, d.y.rows = 4, d.y.cols = 1;
    return d;
}

void test_model(scyte_network* model)
{
    float x_test[2];

    x_test[0]=0.f, x_test[1] = 0.f;
    const float* vals_new = scyte_predict_network(model, x_test);
    printf("%f %f --> %f\n", x_test[0], x_test[1], *vals_new);

    x_test[0]=0.f, x_test[1] = 1.f;
    vals_new = scyte_predict_network(model, x_test);
    printf("%f %f --> %f\n", x_test[0], x_test[1], *vals_new);

    x_test[0]=1.f, x_test[1] = 0.f;
    vals_new = scyte_predict_network(model, x_test);
    printf("%f %f --> %f\n", x_test[0], x_test[1], *vals_new);

    x_test[0]=1.f, x_test[1] = 1.f;
    vals_new = scyte_predict_network(model, x_test);
    printf("%f %f --> %f\n", x_test[0], x_test[1], *vals_new);
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    scyte_data d = generate_xor_data();
#if 1
    double t1 = time_now();
    scyte_network* model = make_model();
    scyte_print_graph(model->n, model->nodes);

    scyte_optimizer_params params = scyte_sgd_params(1.f, 0.0005f, 0.90f);
    scyte_train_network(model, params, 1, 1000, 0, 20, d);
    double t2 = time_now();
    LOG_INFOF("trairning took %.3lf seconds, saving model..", t2-t1);
    scyte_save_network("test.weight", model);
#else
    scyte_network* model = scyte_load_network("test.weight");
    scyte_print_graph(model->n, model->nodes);
#endif
    test_model(model);
    scyte_free_network(model);
    return 0;
}
