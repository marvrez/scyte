#include "optimizer.h"

#include <math.h>

scyte_optimizer_params scyte_adam_params(float lr, float decay, float momentum, float beta_1, float beta_2)
{
    scyte_optimizer_params p;
    p.type = ADAM;
    p.lr = lr;
    p.momentum = momentum;
    p.beta_1 = beta_1, p.beta_2 = beta_2;
    return p;
}

scyte_optimizer_params scyte_rmsprop_params(float lr, float decay, float momentum, float alpha)
{
    scyte_optimizer_params p;
    p.type = RMSPROP;
    p.lr = lr;
    p.momentum = momentum;
    p.alpha = alpha;
    return p;
}

scyte_optimizer_params scyte_sgd_params(float lr, float decay, float momentum)
{
    scyte_optimizer_params p;
    p.type = SGD;
    p.lr = lr;
    p.momentum = momentum;
    return p;
}

#define EPS 1e-6f

void scyte_optimizer_step(scyte_optimizer_params params, int n, const float* g, float* g_prev, float* w)
{
    float lr = params.lr, momentum = params.momentum;
    if(params.type == SGD) {
        for(int i = 0; i < n; ++i) {
            g_prev[i] = -lr*g[i] + momentum*g_prev[i];
            w[i] += g_prev[i];
        }
    }
    else if(params.type == RMSPROP) {
        float alpha = params.alpha; 
        for(int i = 0; i < n; ++i) {
            g_prev[i] = alpha*g_prev[i] + (1.f - alpha)*g[i]*g[i];
            w[i] += -lr*g[i]/sqrtf(g_prev[i] + EPS);
        }
    }
    else if(params.type == ADAM) {
        /*
         * TODO: make a separate step function for each optimizer?
        float beta1 = params.beta_1, beta2 = params.beta_2; 
        for(int i = 0; i < n; ++i) {
            g_mean[i] = beta1*g_prev[i] + (1.f - beta1)*g[i]; // estimate mean of gradient
            g_var[i] = beta2*g_prev[i] + (1.f - beta2)*g[i]*g[i]; // estimate variance of gradients
            w[i] += -lr*g_mean[i]/sqrtf(g_var[i] + EPS);
        }
        */

    }
}
