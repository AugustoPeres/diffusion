import os
from PIL import Image

import numpy as np

import jax.numpy as jnp
import jax.random as jr


def load_images_from_directory(directory,
                               batch_size,
                               target_shape,
                               allowed_formats=('.png', '.jpg', '.jpeg')):
    """
    Load and yield batches of images from a directory.

    Parameters:
    - directory: The path to the directory containing the images.
    - batch_size: The number of images to load and yield at each step.
    - target_shape: The desired shape of the images.
    - allowed_formats: The formats in which the images can be in

    Returns:
    - A generator that yields batches of images.
    """
    image_files = [
        f for f in os.listdir(directory) if f.endswith(allowed_formats)
    ]

    while True:
        batch_files = np.random.choice(image_files, size=batch_size)

        images = []
        for file in batch_files:
            image_path = os.path.join(directory, file)
            image = Image.open(image_path)
            image = jnp.array(
                image, dtype='uint16').reshape(target_shape) * 2 / 255.0 - 1
            images.append(image)

        yield jnp.array(images)
