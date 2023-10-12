from matplotlib import pyplot as plt
import torch

from .fourier import fft_3d


def plot_tomo_slices(tomo, figsize=(7, 5), return_figure=False):
    fig, ax = plt.subplots(2, 3, figsize=figsize)
    # plot in image domain
    half = (torch.tensor(tomo.shape) / 2).floor().int()
    ax[0, 0].imshow(tomo[half[0], :, :], cmap="gray")
    ax[0, 1].imshow(tomo[:, half[1], :], cmap="gray")
    ax[0, 2].imshow(tomo[:, :, half[2]], cmap="gray")
    # plot in Fourier domain
    tomo_ft = fft_3d(tomo).abs()
    ax[1, 0].imshow(tomo_ft[half[0], :, :], cmap="gray")
    ax[1, 1].imshow(tomo_ft[:, half[1], :], cmap="gray")
    ax[1, 2].imshow(tomo_ft[:, :, half[2]], cmap="gray")
    # layout
    ax[0, 0].set_title("Central X-Y Slice")
    ax[0, 1].set_title("Central X-Z Slice")
    ax[0, 2].set_title("Central Y-Z Slice")
    ax[1, 0].set_title("Central X-Y Fourier Slice")
    ax[1, 1].set_title("Central X-Z Fourier Slice")
    ax[1, 2].set_title("Central Y-Z Fourier Slice")
    for a in ax.flatten():
        a.axis("off")
    fig.tight_layout()
    if return_figure:
        return fig
    else:
        fig.show()
