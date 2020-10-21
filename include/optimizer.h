#ifndef OPTIMIZER_H
#define OPTIMIZER_H

typedef enum {
    SGD,
    RMSPROP,
    ADAM,
} scyte_optimizer_type;

typedef struct {
    scyte_optimizer_type type;
    float lr; // learning rate
    float decay; // regularization
    float momentum;
    float alpha; // for RMSProp
    float beta1, beta2; // for adam, exponential averaging
} scyte_optimizer_params;

scyte_optimizer_params scyte_adam_params(float lr, float decay, float momentum, float beta1, float beta2);
scyte_optimizer_params scyte_rmsprop_params(float lr, float decay, float momentum, float alpha);
scyte_optimizer_params scyte_sgd_params(float lr, float decay, float momentum);

void scyte_sgd_step(scyte_optimizer_params params, int n, const float* g, float* g_prev, float* w);
void scyte_rmsprop_step(scyte_optimizer_params params, int n, const float* g, float* g_var, float* w);
void scyte_adam_step(scyte_optimizer_params params, int n, const float* g, float* g_var, float* g_mean, float* w, int t);

#endif
