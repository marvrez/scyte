#ifndef UTILS_H
#define UTILS_H

#include "list.h"
#include <stdio.h>

double time_now();
float randn();
float random_normal(float mu, float sigma);
float random_uniform(float min, float max);
// +1 if positive, -1 if negative, 0 otherwise
int get_sign(float val);

char* get_shape_string(int n, int* shape);

char* fgetl(FILE* fp);
list* read_lines(const char* filename);

 // sorts in increasing value
void qsortf(int n, float* data);

int max_index(const float* a, int n);

#endif
