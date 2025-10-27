import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

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

def plot_tensors(tensor, title=None, cmap="magma", show_colorbar=True, save_path=None):
    """
    Plots a batch of filters/images in a nearly square grid.

    The input tensor should have shape (batch_size, filters, height, width).
    Only the first batch element is visualized.

    Args:
        tensor (torch.Tensor or np.ndarray):
            A 4D tensor or array with shape (B, K, H, W).
        title (str, optional):
            Title for the entire grid plot. Defaults to None.
        cmap (str, optional):
            Colormap for visualization. Defaults to "magma".
        show_colorbar (bool, optional):
            Whether to display a shared colorbar. Defaults to True.
        save_path (str, optional):
            File path to save the plot. If None, the plot is not saved.

    Returns:
        image: PIL image
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    B, K, H, W = tensor.shape
    imgs = tensor[0]  # (K, H, W)

    vmin, vmax = float(imgs.min()), float(imgs.max())

    # --- cuadrado automático ---
    cols = int(np.ceil(np.sqrt(K)))
    rows = int(np.ceil(K / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axs = np.atleast_2d(axs)

    for i in range(K):
        r, c = divmod(i, cols)
        axs[r, c].imshow(imgs[i], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[r, c].axis("off")

    # Apagar celdas sobrantes
    for j in range(K, rows * cols):
        r, c = divmod(j, cols)
        axs[r, c].axis("off")

    # Colorbar compartida
    if show_colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks([vmin, vmax])

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    # plt.show()
    # plt.close()

    image = Image.open(save_path)
    return image