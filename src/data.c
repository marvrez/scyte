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
    m.data = calloc(rows*cols, sizeof(float*));
    return m;
}

void scyte_random_batch(scyte_data d, int batch_size, float* X, float* y)
{
    int x_cols = d.X.cols, y_cols = d.y.cols;
    for(int i = 0; i < batch_size; ++i) {
        int idx = rand() % d.X.rows;
        memcpy(&X[i*x_cols], &d.X.data[idx*x_cols], x_cols*sizeof(float));
        memcpy(&y[i*y_cols], &d.y.data[idx*y_cols], y_cols*sizeof(float));
    }
}

scyte_data load_image_classification_data(char* images, char* label_file, int colored)
{
    list* image_list = read_lines(images);
    list* label_list = read_lines(label_file);
    int n = image_list->size, num_labels = label_list->size, cols = 0, count = 0;
    char** labels = (char**)list_to_array(label_list);
    matrix X, y = make_matrix(n, num_labels);
    list_node* node = image_list->head;
    while(node) {
        char* image_path = (char*)node->data;
        image img = load_image(image_path, !!colored ? 3 : 1);
        if(!cols) {
            cols = img.w*img.h*img.c;
            X = make_matrix(n, cols);
        }
        for(int i = 0; i < cols; ++i) X.data[count*X.cols + i] = img.data[i];
        for(int i = 0; i < num_labels; ++i) {
            if(strstr(image_path, labels[i])) {
                y.data[count*y.cols + i] = 1;
            }
        }
        ++count;
        node = node->next;
    }
    free_list(image_list); free_list(label_list);
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
        for(int j = 0; j < d.X.cols; ++j) printf("%f ", d.X.data[i*d.X.cols + j]);
        printf("--> ");
        for(int j = 0; j < d.y.cols; ++j) printf("%f ", d.y.data[i*d.y.cols + j]);
        printf("\n");
    }
}
