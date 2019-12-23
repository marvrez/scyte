#ifndef DATA_H
#define DATA_H

typedef struct {
    int rows, cols;
    float** data;
} matrix;

typedef struct {
    matrix X;
    matrix y;
} scyte_data;

typedef struct {
    scyte_data* d; // the data structure that will contain the loaded data
    int num_channels; // num channels for image to load
    char** paths; // paths to the data
    char** labels; // labels for the data
    int start_idx; // what idx to start loading the data from
    int end_idx; // what idx to stop loading the data at
} load_args;

void scyte_random_batch(scyte_data d, int batch_size, float* X, float* y);
scyte_data load_image_classification_data(const char* images, const char* label_file, int colored);

void scyte_free_data(scyte_data* d);
void scyte_print_data(scyte_data d);

#endif
