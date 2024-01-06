"""Utils for diffusion training"""
import jax.numpy as jnp


def make_linear_betas(min_beta, max_beta, time_steps):
    """Makes the betas_0, betas_1 for every t using a linear schedule"""
    return jnp.linspace(min_beta, max_beta, time_steps)


def make_alpha_tildas(betas):
    """Makes a list with the alpha_tilda for each time step"""
    alphas = [1 - beta for beta in betas]
    alpha_tildas = []
    current = 1
    for alpha in alphas:
        current *= alpha
        alpha_tildas.append(current)
    return jnp.array(alphas), jnp.array(alpha_tildas)
