#include <stdio.h>

#include "scyte.h"
#include "network.h"
#include "data.h"
#include "utils.h"
#include "logger.h"

#include "arg.h"

scyte_network* make_model_xor()
{
    scyte_node* in = scyte_layer_input(2);
    scyte_node* dense  = scyte_sigmoid(scyte_layer_connected(in, 4));
    scyte_node* loss = scyte_layer_cost(dense, 1, COST_BINARY_CROSS_ENTROPY);
    return scyte_make_network(loss);
}

static inline void train_xor(scyte_network* net, int epochs, float lr, float decay, float momentum)
{
    scyte_data d;
    float* x[4] = { (float[]){ 0, 0 }, (float[]){ 0, 1, }, (float[]){ 1, 0 }, (float[]){ 1, 1 } };
    d.X.data = x; d.X.rows = 4, d.X.cols = 2;
    float* y[4] = { (float[]){ 0 }, (float[]){ 1 }, (float[]){ 1 }, (float[]){ 0 }, };
    d.y.data = y, d.y.rows = 4, d.y.cols = 1;

    scyte_optimizer_params params = scyte_sgd_params(lr, decay, momentum);
    scyte_train_network(net, params, 1, epochs, 0, 20, d);
}

void predict_xor(scyte_network* net)
{
    printf("%f %f --> %f\n", 0.f, 0.f, *scyte_predict_network(net, (float[]){ 0, 0 }));
    printf("%f %f --> %f\n", 0.f, 1.f, *scyte_predict_network(net, (float[]){ 0, 1 }));
    printf("%f %f --> %f\n", 1.f, 0.f, *scyte_predict_network(net, (float[]){ 1, 0 }));
    printf("%f %f --> %f\n", 1.f, 1.f, *scyte_predict_network(net, (float[]){ 1, 1 }));
}

int run_model_xor(int argc, char** argv)
{
    srand(1337);
    int epochs=1000, predict=0, help=0;
    float lr=1.f, momentum=0.9f, decay=0.0005f;

    arg_option_count(&help, 'h', "help", "show this message");
    arg_option_count(&predict, 'p', "predict", "set to use prediction mode, else training mode by default");
    arg_option_int(&epochs, 'e', "epochs", "number of epochs to train model", ARG_REQUIRED);
    arg_option_float(&lr, 'r', "lr", "learning rate for model", ARG_REQUIRED);
    arg_option_float(&momentum, 'm', "momentum", "momentum", ARG_REQUIRED);
    arg_option_float(&decay, 'd', "decay", "l2 decay", ARG_REQUIRED);
    argc = arg_parse(argv);

    if(help) {
        arg_help();
        exit(1);
    }
    if(argc < 3) {
        fprintf(stderr, "Usage: %s %s <model_weight_name> [options]\n", argv[0], argv[1]);
        arg_help();
        exit(1);
    }

    const char* model_path = argv[2];
    scyte_network* model;
    if(predict <= 0) {
        double t1 = time_now();
        model = make_model_xor();
        scyte_print_graph(model->n, model->nodes);
        train_xor(model, epochs, lr, decay, momentum);
        double t2 = time_now();
        LOG_INFOF("training took %.3lf seconds, saving model.. to %s", t2-t1, model_path);
        scyte_save_network(model_path, model);
    }
    else {
        model = scyte_load_network(model_path);
        scyte_print_graph(model->n, model->nodes);
        predict_xor(model);
    }
    scyte_free_network(model);
    return 0;
}
