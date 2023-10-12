import math
import torch

from torch import fft
from .rotation import rotate_vol_around_axis
from .fourier import fft_3d, ifft_3d


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


def get_missing_wedge_mask(grid_size, mw_angle,device="cpu"):
    grid = get_3d_fft_freqs_on_grid(grid_size=grid_size, device=device)
    # make normal vectors of two hyperplanes that bound missing wedge
    alpha = torch.deg2rad(torch.tensor(float(mw_angle)))/2
    normal_left = torch.tensor([torch.sin(alpha), torch.cos(alpha)])
    normal_right = torch.tensor([torch.sin(alpha), -torch.cos(alpha)])
    # # rotate normal vectors
    # rot_angle = torch.deg2rad(torch.tensor(float(rot_angle)))
    # rot_mat = torch.tensor([
    #     [torch.cos(rot_angle), -torch.sin(rot_angle)],
    #     [torch.sin(rot_angle), torch.cos(rot_angle)]
    # ])
    # normal_left = rot_mat @ normal_left
    # normal_right = rot_mat @ normal_right
    # embed normal vectors into x-z plane
    normal_left = torch.tensor([normal_left[0], 0, normal_left[1]], device=device)
    normal_right = torch.tensor([normal_right[0], 0,  normal_right[1]], device=device)
    # select all points that lie above or below both hyperplanes that bound missing wedge
    grid_size = [int(s) for s in grid_size]  # convert to list because reshape needs list or tuple
    upper_wedge = torch.logical_or(grid.inner(normal_left) >= 0, grid.inner(normal_right) >= 0).reshape(list(grid_size))
    lower_wedge = torch.logical_or(grid.inner(normal_left) <= 0, grid.inner(normal_right) <= 0).reshape(list(grid_size))
    mw_mask = torch.logical_and(upper_wedge, lower_wedge).int()
    # finally, mask out everyhting ourside ball that fits inside image
    return mw_mask


def get_rotated_missing_wedge_mask(grid_size, mw_angle, rot_axis, rot_angle, device="cpu"):
    grid_size = torch.tensor(grid_size)
    # enlarge grid size such that rotated grid fits inside
    adjusted_grid_size = (torch.ceil(math.sqrt(2) * grid_size) / 2.) * 2
    mw_mask = get_missing_wedge_mask(grid_size=adjusted_grid_size, mw_angle=mw_angle)
    mw_mask = rotate_vol_around_axis(vol=mw_mask, rot_angle=rot_angle, rot_axis=rot_axis, output_shape=grid_size, order=3).float().to(device)
    return mw_mask