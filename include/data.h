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

void scyte_random_batch(scyte_data d, int batch_size, float* X, float* y);
scyte_data load_image_classification_data(char* images, char* label_file, int colored);

void scyte_free_data(scyte_data* d);
void scyte_print_data(scyte_data d);

#endif
