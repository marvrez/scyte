#include "optimizer.h"

#include <math.h>
#include <assert.h>

#define EPS 1e-6f

scyte_optimizer_params scyte_adam_params(float lr, float decay, float momentum, float beta1, float beta2)
{
    scyte_optimizer_params p;
    p.type = ADAM;
    p.lr = lr;
    p.momentum = momentum;
    p.decay = decay;
    p.beta1 = beta1, p.beta2 = beta2;
    return p;
}

scyte_optimizer_params scyte_rmsprop_params(float lr, float decay, float momentum, float alpha)
{
    scyte_optimizer_params p;
    p.type = RMSPROP;
    p.lr = lr;
    p.momentum = momentum;
    p.alpha = alpha;
    p.decay = decay;
    return p;
}

scyte_optimizer_params scyte_sgd_params(float lr, float decay, float momentum)
{
    scyte_optimizer_params p;
    p.type = SGD;
    p.lr = lr;
    p.decay = decay;
    p.momentum = momentum;
    return p;
}

void scyte_sgd_step(scyte_optimizer_params params, int n, const float* g, float* g_prev, float* w)
{
    assert(params.type == SGD);
    float lr = params.lr, momentum = params.momentum, decay = params.decay;
    for(int i = 0; i < n; ++i) {
        g_prev[i] = g[i] - momentum*g_prev[i] + decay*w[i];
        w[i] -= lr*g_prev[i];
    }
}

void scyte_rmsprop_step(scyte_optimizer_params params, int n, const float* g, float* g_var, float* w)
{
    assert(params.type == RMSPROP);
    float lr = params.lr, alpha = params.alpha, decay = params.decay;
    for(int i = 0; i < n; ++i) {
        g_var[i] = alpha*g_var[i] + (1.f - alpha)*g[i]*g[i]; // estimate variance of gradients
        w[i] -= lr*(g[i]/sqrtf(g_var[i] + EPS) + decay*w[i]);
    }
}

void scyte_adam_step(scyte_optimizer_params params, int n, const float* g, float* g_var, float* g_mean, float* w, int t)
{
    assert(params.type == ADAM);
    float lr = params.lr, beta1 = params.beta1, beta2 = params.beta2, decay = params.decay;
    for(int i = 0; i < n; ++i) {
        g_mean[i] = beta1*g_mean[i] + (1.f - beta1)*g[i]; // estimate mean of gradient
        g_var[i] = beta2*g_var[i] + (1.f - beta2)*g[i]*g[i]; // estimate variance of gradients
        // bias correction of raw moments
        float mhat = g_mean[i] / (1.f - powf(beta1, t));
        float vhat = g_var[i] / (1.f - powf(beta2, t));
        w[i] -= lr*(mhat/sqrtf(vhat + EPS) + decay*w[i]);
    }
}
