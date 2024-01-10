"""Visualization functions"""
import matplotlib.pyplot as plt


def plot_image(image, data_shape, save_path):
    reshape_to = tuple(reversed(data_shape))
    fig, ax = plt.subplots()
    ax.imshow(image.reshape(reshape_to), cmap='gray')
    cbar = plt.colorbar(ax.imshow(image.reshape(reshape_to), cmap='gray'),
                        ax=ax)
    plt.savefig(save_path)
