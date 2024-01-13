import os
from PIL import Image

import numpy as np

import jax.numpy as jnp
import jax.random as jr


def load_images_from_directory(directory,
                               batch_size,
                               allowed_formats=('.png', '.jpg', '.jpeg'),
                               black_and_white=True):
    """
    Load and yield batches of images from a directory.

    Parameters:
    - directory: The path to the directory containing the images.
    - batch_size: The number of images to load and yield at each step.
    - allowed_formats: The formats in which the images can be in
    - black_and_white: Controls if the image is black and white or not.

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
            if not black_and_white:
                image = image.convert('RGB')
            image = jnp.array(image, dtype='uint16')
            if not black_and_white:
                image = jnp.transpose(image, (2, 0, 1))
            image = image * 2 / 255.0 - 1
            images.append(image)

        yield jnp.array(images)
