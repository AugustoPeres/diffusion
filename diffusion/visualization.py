"""Visualization functions"""
import matplotlib.pyplot as plt


def plot_image(image, save_path):
    fig, ax = plt.subplots()
    ax.imshow(image.reshape(28, 28),
              cmap='gray')  # You can choose a different colormap if needed
    cbar = plt.colorbar(ax.imshow(image.reshape(28, 28), cmap='gray'), ax=ax)
    plt.savefig(save_path)
