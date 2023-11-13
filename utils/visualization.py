from matplotlib import pyplot as plt
import torch

from .fourier import fft_3d


def plot_tomo_slices(tomo, domain="image", figsize=(7, 5)):
    # docstring
    """
    Plot central slices of a 3D tomogram in either image or Fourier domain.
    tomo: 3D tensor, tomogram to plot
    domain: "image" or "fourier", domain to plot in
    figsize: tuple of two integers, size of the figure
    """
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    # plot in image domain
    half = (torch.tensor(tomo.shape) / 2).floor().int()
    if domain == "image":
        ax[0].imshow(tomo[half[0], :, :], cmap="gray")
        ax[1].imshow(tomo[:, half[1], :], cmap="gray")
        ax[2].imshow(tomo[:, :, half[2]], cmap="gray")
    elif domain == "fourier":
        # plot in Fourier domain
        tomo_ft = fft_3d(tomo).abs()
        ax[0].imshow(tomo_ft[half[0], :, :], cmap="gray")
        ax[1].imshow(tomo_ft[:, half[1], :], cmap="gray")
        ax[2].imshow(tomo_ft[:, :, half[2]].T, cmap="gray")
    # layout
    ax[1].set_title(f"Central {'Fourier' if domain=='fourier' else ''} X-Z Slice")
    ax[2].set_title(f"Central {'Fourier' if domain=='fourier' else ''} Y-Z Slice")
    ax[0].set_title(f"Central {'Fourier' if domain=='fourier' else ''} X-Y Slice")
    for a in ax.flatten():
        a.axis("off")
    fig.tight_layout()
    return fig
