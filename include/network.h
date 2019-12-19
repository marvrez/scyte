#ifndef NETWORK_H
#define NETWORK_H

#include "scyte.h"
#include "op.h"
#include "layers.h"
#include "optimizer.h"
#include "data.h"

// Generates a network from a computational graph.
// A network must have at least one scalar cost node (i.e. whose num_dims==0).
scyte_network* scyte_make_network(scyte_node* cost_node);
// create a network from multiple root nodes.
scyte_network* scyte_make_network2(scyte_node* cost_node, int n_roots, scyte_node** roots);
void scyte_free_network(scyte_network* net);

void scyte_train_network(scyte_network* net, scyte_optimizer_params params, int batch_size, int num_epochs, float val_split, int early_stop_patience, scyte_data data);
const float* scyte_predict_network(scyte_network* net, float* data);

void scyte_save_network(const char* filename, scyte_network* net);
scyte_network* scyte_load_network(const char* filename);

#endif
