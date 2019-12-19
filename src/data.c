#include "data.h"

#include "list.h"
#include "image.h"
#include "utils.h"
#include "logger.h"

#include <pthread.h>
#include <unistd.h>

#ifdef _SC_NPROCESSORS_ONLN
#define NUM_THREADS sysconf(_SC_NPROCESSORS_ONLN)
#else
#define NUM_THREADS 4
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

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

void* load_classification_data(void* args)
{
    load_args* largs = (load_args*)args;
    int start_idx = largs->start_idx, end_idx = largs->end_idx;
    for(int i = start_idx; i < end_idx; ++i) {
        char* image_path = largs->paths[i];
        image img = load_image(image_path, largs->num_channels);
        largs->d->X.data[i] = img.data, largs->d->X.cols = img.w*img.h*img.c;
        for(int j = 0; j < largs->d->y.cols; ++j) {
            if(strstr(image_path, largs->labels[j])) {
                largs->d->y.data[i][j] = 1;
            }
        }
    }
    return NULL;
}

scyte_data load_image_classification_data(char* images, char* label_file, int colored)
{
    int num_channels = !!colored ? 3 : 1;
    list* image_list = read_lines(images), *label_list = read_lines(label_file);
    char** labels = (char**)list_to_array(label_list), **paths = (char**)list_to_array(image_list);
    int n = image_list->size, num_labels = label_list->size;
    matrix X = make_matrix(n, 1), y = make_matrix(n, num_labels);
    scyte_data d = { X, y };

    pthread_t* threads = calloc(NUM_THREADS, sizeof(pthread_t));
    load_args* args = calloc(NUM_THREADS, sizeof(load_args));
    for(int i = 0; i < NUM_THREADS; ++i) {
        args[i].d = &d, args->num_channels = num_channels, args[i].paths = paths, args[i].labels = labels;
        args[i].start_idx = i*n/NUM_THREADS, args[i].end_idx = (i+1)*n/NUM_THREADS;
        int error = pthread_create(&threads[i], 0, load_classification_data, &args[i]);
        if(error) {
            LOG_ERRORF("failed to create thread, error code: %d", error);
            assert(0);
        }
    }
    for(int i = 0; i < NUM_THREADS; ++i) pthread_join(threads[i], 0);
    free_list(image_list); free_list(label_list);
    free(paths); free(labels); free(args); free(threads);
    return d;
}

void scyte_free_data(scyte_data* d)
{
    for(int i = 0; i < d->X.rows; ++i) free(d->X.data[i]);
    for(int i = 0; i < d->y.rows; ++i) free(d->y.data[i]);
    free(d->X.data), free(d->y.data);
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
