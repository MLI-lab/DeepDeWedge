import torch

from .fourier import apply_fourier_mask_to_tomo


def masked_loss(model_output, target, rot_mw_mask, mw_mask, mw_weight=2.0):
    """
    The self-supervised per-sample loss function for denoising and missing wedge reconstruction.
    """
    outside_mw_mask = rot_mw_mask * mw_mask
    outside_mw_loss = (
        apply_fourier_mask_to_tomo(
            tomo=target - model_output, mask=outside_mw_mask, output="real"
        )
        .abs()
        .pow(2)
        .mean()
    )
    inside_mw_mask = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
    inside_mw_loss = (
        apply_fourier_mask_to_tomo(
            tomo=target - model_output, mask=inside_mw_mask, output="real"
        )
        .abs()
        .pow(2)
        .mean()
    )
    loss = outside_mw_loss + mw_weight * inside_mw_loss
    return loss
