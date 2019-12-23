#include <stdio.h>

#include "scyte.h"
#include "op.h"
#include "network.h"
#include "data.h"
#include "image.h"

#include "arg.h"
#include "logger.h"
#include "utils.h"

scyte_network* make_model_mnist()
{
    scyte_node* t;
    t = scyte_layer_input_image(28,28,1);
    t = scyte_relu(scyte_layer_conv2d(t, 8, 3, 1, 0));
    t = scyte_relu(scyte_layer_conv2d(t, 16, 3, 2, 0));
    t = scyte_relu(scyte_layer_conv2d(t, 32, 3, 2, 0));
    t = scyte_relu(scyte_layer_conv2d(t, 32, 3, 2, 0));
    scyte_node* loss = scyte_layer_cost(t, 10, COST_CROSS_ENTROPY);
    return scyte_make_network(loss);
}

int run_model_mnist(int argc, char** argv)
{
    srand(1337);
    int epochs=1000, batch_size=256, predict=0, help=0;
    float lr=0.01f, momentum=0.9f, decay=0.0005f;

    const char* input_image_path = 0;
    const char* labels_path = 0;
    const char* data_path = 0;

    arg_option_count(&help, 'h', "help", "show this message");
    arg_option_count(&predict, 'p', "predict", "set to use prediction mode, else training mode by default");
    arg_option_string(&labels_path, 'l', "label_path", "path to file containing labels", ARG_REQUIRED);
    arg_option_string(&data_path, 0, "data_path", "path to file containing data paths", ARG_REQUIRED);
    arg_option_string(&input_image_path, 'i', "input_image", "input image path for prediction", ARG_REQUIRED);
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
    if(!predict) {
        double a1 = time_now();
        scyte_data d = load_image_classification_data(data_path, labels_path, 0);
        double a2 = time_now();
        LOG_INFOF("loading data took %.3lf seconds", a2-a1);

        double t1 = time_now();
        model = make_model_mnist();
        scyte_print_graph(model->n, model->nodes);

        scyte_optimizer_params params = scyte_sgd_params(lr, decay, momentum);
        scyte_train_network(model, params, batch_size, epochs, 0.2, 10, d);
        double t2 = time_now();
        LOG_INFOF("training took %.3lf seconds, saving model..", t2-t1);
        scyte_save_network(model_path, model);
    }
    else {
        model = scyte_load_network(model_path);
        scyte_print_graph(model->n, model->nodes);

        image img = load_image_grayscale(input_image_path);
        const float* vals = scyte_predict_network(model, img.data);
        int max_idx = max_index(vals, 10);
        LOG_INFOF("input image '%s' was predicted as the number %d with probability %.2f\n", input_image_path, max_idx, vals[max_idx]);
    }

    scyte_free_network(model);
    return 0;
}
