import os
import json

from absl import app
from absl import flags

import jax.random as jr
import equinox as eqx

from diffusion.models import UNet
import diffusion.training as training
import diffusion.utils as diffusion_utils
import diffusion.data as data

import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('path_to_data', None, 'The path to the data.')
flags.DEFINE_string('output_dir', None, 'Save artifacts of the run.')

flags.DEFINE_list('data_shape', [1, 28, 28], 'Shape of the data.')
flags.DEFINE_boolean('is_biggan', False, 'Whether the model is big gan.')
flags.DEFINE_list('dim_mults', [4, 4], 'dim_mults list.')
flags.DEFINE_integer('hidden_size', 64, 'Size of hidden layers.')
flags.DEFINE_integer('heads', 4, 'The attention heads.')
flags.DEFINE_integer('dim_head', 16, 'Dimensionality of the attention heads.')
flags.DEFINE_float('dropout_rate', 0.1, 'The dropout rate.')
flags.DEFINE_integer('num_res_blocks', 2, 'The number of residual blocks.')
flags.DEFINE_list('attn_resolutions', [4, 4],
                  'The resolution of the attention.')

flags.DEFINE_float('learning_rate', 1e-4, 'The learning rate of the model.')
flags.DEFINE_integer('batch_size', 16, 'The batch size for training.')
flags.DEFINE_integer('num_diffusion_steps', 1000,
                     'The number of diffusion steps.')
flags.DEFINE_float('min_beta', 10e-4, 'The minimum value of beta.')
flags.DEFINE_float('max_beta', 0.02, 'The maximum value of beta.')
flags.DEFINE_integer('num_training_iterations', 1000,
                     'The number of training iterations.')
flags.DEFINE_integer('logging_frequency', 50, 'After how many steps to log.')

flags.DEFINE_integer('num_samples', 5,
                     'Number of images to sample after training.')
flags.DEFINE_integer('storing_frequency', 10,
                     'Frequency to store images during sampling')

flags.DEFINE_integer('random_key', 0, 'The random key.')


def save_flags_to_json(flags_dict, file_path):
    """
    Save flags from a dictionary to a JSON file.

    Parameters:
    - flags_dict (dict): Dictionary containing the flags to be saved.
    - file_path (str): Path to the JSON file where the flags will be saved.
    """
    try:
        directory = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'w') as json_file:
            json.dump(flags_dict, json_file, indent=4)
        print(f'Flags saved successfully to {file_path}')
    except Exception as e:
        print(f'Error saving flags to {file_path}: {e}')


def main(_):
    key = jr.PRNGKey(FLAGS.random_key)

    save_flags_to_json(
        {
            'data_shape': FLAGS.data_shape,
            'is_biggan': FLAGS.is_biggan,
            'dim_mults': FLAGS.dim_mults,
            'hidden_size': FLAGS.hidden_size,
            'heads': FLAGS.heads,
            'dim_head': FLAGS.dim_head,
            'dropout_rate': FLAGS.dropout_rate,
            'num_res_blocks': FLAGS.num_res_blocks,
            'attn_resolutions': FLAGS.attn_resolutions,
            'learning_rate': FLAGS.learning_rate,
            'batch_size': FLAGS.batch_size,
            'num_diffusion_steps': FLAGS.num_diffusion_steps,
            'min_beta': FLAGS.min_beta,
            'max_beta': FLAGS.max_beta,
            'num_training_iterations': FLAGS.num_training_iterations,
            'logging_frequency': FLAGS.logging_frequency,
            'num_samples': FLAGS.num_samples,
            'storing_frequency': FLAGS.storing_frequency,
            'random_key': FLAGS.random_key,
        }, os.path.join(FLAGS.output_dir, 'flags.json'))

    model_key, training_key, sampling_key = jr.split(key, 3)

    dataloader = data.load_images_from_directory(
        FLAGS.path_to_data, FLAGS.batch_size, tuple(map(int,
                                                        FLAGS.data_shape)))

    model = UNet(data_shape=tuple(map(int, FLAGS.data_shape)),
                 is_biggan=FLAGS.is_biggan,
                 dim_mults=list(map(int, FLAGS.dim_mults)),
                 hidden_size=FLAGS.hidden_size,
                 heads=FLAGS.heads,
                 dim_head=FLAGS.dim_head,
                 dropout_rate=FLAGS.dropout_rate,
                 num_res_blocks=FLAGS.num_res_blocks,
                 attn_resolutions=list(map(int, FLAGS.attn_resolutions)),
                 key=model_key)

    betas = diffusion_utils.make_linear_betas(FLAGS.min_beta, FLAGS.max_beta,
                                              FLAGS.num_diffusion_steps)
    alphas, alpha_tildas = diffusion_utils.make_alpha_tildas(betas)

    model = training.train_model(model, 0, FLAGS.num_diffusion_steps,
                                 alpha_tildas, dataloader, training_key,
                                 FLAGS.learning_rate,
                                 FLAGS.num_training_iterations,
                                 FLAGS.logging_frequency)

    eqx.tree_serialise_leaves(os.path.join(FLAGS.output_dir, 'model.eqx'),
                              model)

    utils.make_samples(model, sampling_key, FLAGS.num_samples,
                       tuple(map(int, FLAGS.data_shape)), betas, alphas,
                       alpha_tildas, FLAGS.num_diffusion_steps,
                       FLAGS.storing_frequency, FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)
