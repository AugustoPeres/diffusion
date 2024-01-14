"""Model training functions"""
import os
import functools

import jax
import jax.numpy as jnp
import jax.random as jr

import optax

import equinox as eqx


@eqx.filter_jit
def add_noise_to_image(alphas, example, t, key):
    alpha_t = alphas[t]
    noise = jr.normal(key=key, shape=example.shape)
    return jnp.sqrt(alpha_t) * example + jnp.sqrt(1 - alpha_t) * noise, noise


@eqx.filter_jit
def single_example_loss(model, t_min, t_max, alphas, example, key):
    """
    Args:
      model: The mode to be trained
      t_min: minimum time_step
      t_max: max time_step
      alphas: List with the values of alpha_tilde_t
      example: A sample from the dataset
    """
    noise_key, model_key, t_key = jr.split(key, 3)
    t = jr.choice(a=jnp.array(range(t_min, t_max)), shape=(), key=t_key)
    model_input, noise = add_noise_to_image(alphas, example, t, noise_key)
    noise_prediction = model(t=t, y=model_input, key=model_key)
    return jnp.mean((noise - noise_prediction)**2)


@eqx.filter_jit
def batch_loss(model, t_min, t_max, alphas, batch, key):
    keys = jr.split(key, batch.shape[0])
    loss_fn = functools.partial(single_example_loss, model, t_min, t_max,
                                jnp.array(alphas))
    loss_fn = jax.vmap(loss_fn)
    return jnp.mean(loss_fn(batch, keys))


@eqx.filter_jit
def model_training_step(model, t_min, t_max, alphas, batch, key,
                        optimizer_state, optimizer):
    loss, grads = eqx.filter_value_and_grad(batch_loss)(model, t_min, t_max,
                                                        alphas, batch, key)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, optimizer_state


def train_model(model,
                t_min,
                t_max,
                alphas,
                dataloader,
                key,
                learning_rate,
                num_steps,
                accumulate_batches,
                log_frequency=50,
                output_dir=None,
                model_logging_frequency=100):
    optimizer = optax.adam(learning_rate)
    if accumulate_batches > 1:
        optimizer = optax.MultiSteps(optimizer,
                                     every_k_schedule=accumulate_batches)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    for i, batch in zip(range(num_steps), dataloader):
        loss, model, key, optimizer_state = model_training_step(
            model, t_min, t_max, alphas, batch, key, optimizer_state,
            optimizer)

        if i % log_frequency == 0:
            print(f'loss at step {i} = {loss}')

        if output_dir is not None:
            if i % model_logging_frequency == 0:
                eqx.tree_serialise_leaves(
                    os.path.join(output_dir, f'model_step_{i}.eqx'), model)
                print(f'saving model at step {i} with loss {loss}')

    return model
