#include "utils.h"

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define M_TWO_PI 6.2831853071795864769252866f

double time_now() 
{
    struct timeval time;
    if (gettimeofday(&time,NULL)) return 0;
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}

// From https://en.wikipedia.org/wiki/Box-Muller_transform
float randn()
{
    static int have_spare = 0;
    static double rand1, rand2;

    if(have_spare) {
        have_spare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    have_spare = 1;

    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * M_TWO_PI;

    return sqrt(rand1) * cos(rand2);
}

float random_normal(float mu, float sigma)
{
    return mu + sigma*randn();
}

float random_uniform(float min, float max)
{
    if(max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

int get_sign(float val)
{
    return (val > 0) - (val < 0);
}

char* get_shape_string(int n, int* shape)
{
    int i;
    if(shape == NULL || n <= 0) return NULL;
    for(i = 0; i < n && shape[i] <= 0; ++i) {}
    char* ret = malloc(256*sizeof(char)), tmp[32];
    sprintf(tmp, "(%d", shape[i++]);
    strcat(ret, tmp);
    for(; i < n; ++i) {
        memset(tmp, 0, sizeof(tmp));
        if(shape[i] < 0) strcpy(tmp,",newaxis");
        else sprintf(tmp, ",%d", shape[i]);
        strcat(ret, tmp);
    }
    strcat(ret, ")");
    return ret;
}

int qsortf_cmp(const void* a, const void* b)
{
    float af = *(const float*)a;
    float bf = *(const float*)b;
    if (af < bf) return -1;
    if (af > bf) return +1;
    return 0;
}

void qsortf(int n, float* data)
{
    qsort(data, n, sizeof(float), qsortf_cmp);
}
