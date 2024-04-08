import torch
from matplotlib import pyplot as plt

from .fourier import fft_3d


def plot_tomo_slices(tomo, domain="image", figsize=(7, 5)):
    # docstring
    """
    Plot central slices of a 3D tomogram in either image or Fourier domain.
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


# def plot_to_tensorboard(writer, fig, tag, step):
#     """
#     Takes a matplotlib figure handle and converts it using
#     canvas and string-casts to a numpy array that can be
#     visualized in TensorBoard using the add_image function

#     Parameters:
#         writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
#         fig (matplotlib.pyplot.fig): Matplotlib figure handle.
#         step (int): counter usually specifying steps/epochs/time.
#     """

#     # Draw figure on canvas
#     fig.canvas.draw()

#     # Convert the figure to numpy array, read the pixel values and reshape the array
#     img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose()
#     img = np.swapaxes(img, 2, 1)  # otherwise image is transposed

#     # Normalize into 0-1 range for TensorBoard(vol). Swap axes for newer versions where API expects colors in first dim
#     img = img / 255.0
#     # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

#     # Add figure in numpy "image" to TensorBoard writer
#     writer.add_image(tag, img, step)
#     plt.close(fig)
