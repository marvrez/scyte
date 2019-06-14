#ifndef UTILS_H
#define UTILS_H

double time_now();
float randn();
float random_normal(float mu, float sigma);
float random_uniform(float min, float max);

char* get_shape_string(int n, int* shape);

 // sorts in increasing value
void qsortf(int n, float* data);

#endif
