import os

import jax.random as jr

import diffusion.visualization as visualization
import diffusion.sampling as sampling


def make_samples(model, key, num_samples, shape, betas, alphas, alpha_tildas,
                 time_steps, storing_frequency, save_path, black_and_white):
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
            save_image_path = os.path.join(save_dir_image, f'image_{j:05}.png')
            if not os.path.exists(save_dir_image):
                os.mkdir(save_dir_image)
            visualization.plot_image(image,
                                     save_image_path,
                                     black_and_white=black_and_white)
