#include "data.h"

#include "list.h"
#include "image.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static inline matrix make_matrix(int rows, int cols)
{
    matrix m;
    m.rows = rows, m.cols = cols;
    m.data = calloc(rows, sizeof(float*));
    for(int i = 0; i < m.rows; ++i){
        m.data[i] = calloc(m.cols, sizeof(float));
    }
    return m;
}

void scyte_random_batch(scyte_data d, int batch_size, float* X, float* y)
{
    int x_cols = d.X.cols, y_cols = d.y.cols;
    for(int i = 0; i < batch_size; ++i) {
        int idx = rand() % d.X.rows;
        memcpy(&X[i*x_cols], d.X.data[idx], x_cols*sizeof(float));
        memcpy(&y[i*y_cols], d.y.data[idx], y_cols*sizeof(float));
    }
}

scyte_data load_image_classification_data(char* images, char* label_file, int colored)
{
    int num_channels = !!colored ? 3 : 1;
    list* image_list = read_lines(images), *label_list = read_lines(label_file);
    char** labels = (char**)list_to_array(label_list), **paths = (char**)list_to_array(image_list);
    int n = image_list->size, num_labels = label_list->size;
    matrix X = make_matrix(n, 1), y = make_matrix(n, num_labels);
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        char* image_path = paths[i];
        image img = load_image(image_path, num_channels);
        X.data[i] = img.data, X.cols = img.w*img.h*img.c;
        for(int j = 0; j < num_labels; ++j) {
            if(strstr(image_path, labels[j])) {
                y.data[i][j] = 1;
            }
        }
    }
    free_list(image_list); free_list(label_list);
    free(paths); free(labels);
    scyte_data d = { X, y };
    return d;
}

void scyte_free_data(scyte_data* d)
{
    free(d->X.data);
    free(d->y.data);
}

void scyte_print_data(scyte_data d)
{
    for(int i = 0; i < d.X.rows; ++i) {
        for(int j = 0; j < d.X.cols; ++j) printf("%f ", d.X.data[i][j]);
        printf("--> ");
        for(int j = 0; j < d.y.cols; ++j) printf("%f ", d.y.data[i][j]);
        printf("\n");
    }
}
