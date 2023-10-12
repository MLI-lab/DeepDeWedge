import torch
from torch import fft


def fft_3d(tomo, norm="ortho"):
    fft_dim = (-1, -2, -3)
    return fft.fftshift(fft.fftn(tomo, dim=fft_dim, norm=norm), dim=fft_dim)


def ifft_3d(tomo, norm="ortho"):
    fft_dim = (-1, -2, -3)
    return fft.ifftn(fft.ifftshift(tomo, dim=fft_dim), dim=fft_dim, norm=norm)


def apply_fourier_mask_to_tomo(tomo, mask, output="real"):
    tomo_ft = fft_3d(tomo)
    tomo_ft_masked = tomo_ft * mask
    vol_filt = ifft_3d(tomo_ft_masked)
    if output == "real":
        return vol_filt.real
    elif output == "complex":
        return vol_filt
    

def get_3d_fft_freqs_on_grid(grid_size, device="cpu"):
    z = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[0]), device=device))
    y = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[1]), device=device))
    x = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[2]), device=device))
    grid = torch.cartesian_prod(z, y, x)
    return grid