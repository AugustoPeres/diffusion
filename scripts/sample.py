"""Sample images using a given model."""
import os
import json

from absl import app
from absl import flags

import jax.random as jr
import equinox as eqx

from diffusion.models import UNet
import diffusion.utils as diffusion_utils

import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('path_to_flags', None, 'Path to a json containing flags.')
flags.DEFINE_string('path_to_weights', None,
                    'Path to the trained model weights.')

flags.DEFINE_integer('num_samples', 5,
                     'Number of images to sample after training.')
flags.DEFINE_integer('storing_frequency', 10,
                     'Frequency to store images during sampling')

flags.DEFINE_integer('random_key', 0, 'The random key.')

flags.DEFINE_string('output_dir', None, 'Path to save the images to.')


def main(_):
    key = jr.PRNGKey(FLAGS.random_key)
    model_key, sampling_key = jr.split(key, 2)

    # load the model here
    with open(FLAGS.path_to_flags, 'r', encoding='utf-8') as f:
        flags = json.load(f)

    data_shape = flags['data_shape']
    num_diffusion_steps = flags['num_diffusion_steps']

    betas = diffusion_utils.make_linear_betas(flags['min_beta'],
                                              flags['max_beta'],
                                              num_diffusion_steps)
    alphas, alpha_tildas = diffusion_utils.make_alpha_tildas(betas)

    model = UNet(
        data_shape=data_shape,
        is_biggan=flags['is_biggan'],
        dim_mults=flags['dim_mults'],
        hidden_size=flags['hidden_size'],
        heads=flags['heads'],
        dim_head=flags['dim_head'],
        dropout_rate=0,  # Use zero dropout since we are in inference.
        num_res_blocks=flags['num_res_blocks'],
        attn_resolutions=flags['attn_resolutions'],
        key=model_key,
    )

    model = eqx.tree_deserialise_leaves(FLAGS.path_to_weights, model)

    utils.make_samples(model, sampling_key, FLAGS.num_samples, data_shape,
                       betas, alphas, alpha_tildas, num_diffusion_steps,
                       FLAGS.storing_frequency, FLAGS.output_dir,
                       flags['black_and_white'])


if __name__ == '__main__':
    app.run(main)
