import matplotlib.pyplot as plt
import numpy as np
import torch
from io import BytesIO
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

def plot_tensors(
    tensor,
    title=None,
    cmap="magma",
    show_colorbar=True,
    save_path=None,
    max_cols=None,
    tile_size=2.0,
):
    """
    Plots a batch of filters/images in a nearly square grid.

    The input tensor should have shape (batch_size, filters, height, width).
    All elements in the batch are visualized.

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
        max_cols (int, optional):
            Maximum number of columns in the grid. Defaults to a square layout.
        tile_size (float, optional):
            Size (in inches) of each subplot tile. Defaults to 2.0.

    Returns:
        image: PIL image
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    B, K, H, W = tensor.shape
    imgs = tensor.reshape(B * K, H, W)
    N = B * K

    # --- cuadrado automático ---
    if max_cols is None:
        cols = int(np.ceil(np.sqrt(N)))
    else:
        cols = min(max_cols, N)
    cols = max(cols, 1)
    rows = int(np.ceil(N / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * tile_size, rows * tile_size))
    axs = np.atleast_2d(axs)

    global_vmin = float(imgs.min())
    global_vmax = float(imgs.max())

    for idx in range(N):
        row, col = divmod(idx, cols)
        ax = axs[row, col]
        vmin, vmax = float(imgs[idx].min()), float(imgs[idx].max())
        ax.imshow(imgs[idx], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        batch_idx = idx // K
        filter_idx = idx % K
        ax.set_title(f"Filter {filter_idx}, batch {batch_idx}", fontsize=10)

    # Apagar celdas sobrantes
    for idx in range(N, rows * cols):
        r, c = divmod(idx, cols)
        axs[r, c].axis("off")

    # Colorbar compartida
    if show_colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        if np.isclose(global_vmin, global_vmax):
            eps = 1e-6 if global_vmin == 0 else abs(global_vmin) * 1e-6
            norm = Normalize(vmin=global_vmin - eps, vmax=global_vmax + eps)
            cbar_ticks = [global_vmin]
        else:
            norm = Normalize(vmin=global_vmin, vmax=global_vmax)
            cbar_ticks = [global_vmin, global_vmax]
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks(cbar_ticks)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)

    image = Image.open(buf)
    if save_path:
        image.save(save_path, format="PNG")

    image_copy = image.copy()
    image.close()
    buf.close()
    plt.close(fig)

    return image_copy
