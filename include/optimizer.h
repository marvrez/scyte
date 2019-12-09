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
    float beta_1, beta_2; // for adam, exponential averaging
} scyte_optimizer_params;

scyte_optimizer_params scyte_adam_params(float lr, float decay, float momentum, float beta_1, float beta_2);
scyte_optimizer_params scyte_rmsprop_params(float lr, float decay, float momentum, float alpha);
scyte_optimizer_params scyte_sgd_parmas(float lr, float decay, float momentum);

void scyte_optimizer_step(scyte_optimizer_params params, int n, const float* g, float* out, float* mem);

#endif
