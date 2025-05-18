import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_tensor(tensor, title=None, save=False):
    tensor = tensor.detach().cpu().numpy()
    vmin, vmax = tensor.min(), tensor.max()
    cmap = 'magma'     # Puedes cambiar a 'magma'

    plt.figure(figsize=(4, 4))
    im = plt.imshow(tensor, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    
    if title:
        plt.title(title)

    # Crear colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks([vmin, vmax])

    if save and title:
        plt.savefig(title + '.png', bbox_inches='tight')

    plt.show()
    plt.close()

def plot_tensors(tensor, title=None):
    """"
    Plot a batch of images in a grid. Creates a grid of filters from a tensor.
    The input tensor should be of shape (batch_size, filters, height, width).
    """
    batch, filters, height, width = tensor.shape
    fig, axs = plt.subplots(filters // width, width, figsize=(width + 2.5 , (filters // width) ))

    vmin = tensor.min()
    vmax = tensor.max()
    cmap = 'magma'

    
    for i in range(filters):
        ax = axs[i // width, i % width]
        ax.imshow(tensor[0, i].cpu().numpy(), cmap)
        ax.axis('off')

    # Agrega colorbar
    cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([vmin, vmax])


    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()
    plt.close()