import os

from absl import app
from absl import flags

import jax.random as jr

from diffusion.models import UNet
import diffusion.training as training
import diffusion.utils as utils
import diffusion.data as data
import diffusion.visualization as visualization
import diffusion.sampling as sampling

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


def make_samples(model, key, num_samples, shape, betas, alphas, alpha_tildas,
                 time_steps, storing_frequency, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in range(num_samples):
        noise_key, key = jr.split(key)
        noise = jr.normal(key=noise_key, shape=shape)
        sampled_images = sampling.sample(model,
                                         noise,
                                         betas,
                                         alphas,
                                         alpha_tildas,
                                         time_steps,
                                         key,
                                         storing_frequency=storing_frequency)
        for j, image in enumerate(sampled_images):
            save_dir_image = os.path.join(save_path, f'samples_{i}')
            save_image_path = os.path.join(save_dir_image, f'image_{j}.png')
            if not os.path.exists(save_dir_image):
                os.mkdir(save_dir_image)
            visualization.plot_image(image, save_image_path)


def main(_):
    key = jr.PRNGKey(FLAGS.random_key)
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

    betas = utils.make_linear_betas(FLAGS.min_beta, FLAGS.max_beta,
                                    FLAGS.num_diffusion_steps)
    alphas, alpha_tildas = utils.make_alpha_tildas(betas)

    model = training.train_model(model, 0, FLAGS.num_diffusion_steps,
                                 alpha_tildas, dataloader, training_key,
                                 FLAGS.learning_rate,
                                 FLAGS.num_training_iterations,
                                 FLAGS.logging_frequency)

    make_samples(model, sampling_key, FLAGS.num_samples,
                 tuple(map(int, FLAGS.data_shape)), betas, alphas,
                 alpha_tildas, FLAGS.num_diffusion_steps,
                 FLAGS.storing_frequency, FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)
