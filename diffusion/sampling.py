"""Sampling functions for the diffusion models"""
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx


def sample(model,
           initial_noise,
           betas,
           alphas,
           alpha_tildas,
           time_step,
           key,
           storing_frequency=50):
    x = initial_noise
    shape = x.shape
    next_key = key
    images = []
    for t in reversed(range(time_step)):
        noise_key, model_key, next_key = jr.split(next_key, 3)
        x = sampling_step(model, alphas, alpha_tildas, betas, x, t, shape,
                          model_key, noise_key)
        if t % storing_frequency == 0:
            images.append(x)
            print(f'denoised time step = {t}')
    return images


# @eqx.filter_jit
def sampling_step(model, alphas, alpha_tildas, betas, x, t, shape, model_key,
                  noise_key):
    predicted_noise = model(t, x, key=model_key)
    x -= ((1 - alphas[t]) / (jnp.sqrt(1 - alpha_tildas[t]))) * predicted_noise
    x /= jnp.sqrt(alphas[t])
    z = jr.normal(key=noise_key, shape=shape) if t > 0 else 0
    x += jnp.sqrt(betas[t]) * z
    return x
