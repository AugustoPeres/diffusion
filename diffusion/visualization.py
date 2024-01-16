"""Visualization functions"""
import jax.numpy as jnp

import matplotlib.pyplot as plt


def plot_image(image,
               save_path=None,
               input_transform=lambda x: (x + 1) / 2,
               black_and_white=True,
               axis='off'):
    image = input_transform(image)
    if black_and_white:
        image = image[0]
    else:
        image = jnp.transpose(image, (1, 2, 0))
    fig, ax = plt.subplots()
    ax.axis(axis)
    if black_and_white:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image, cmap='gray')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
