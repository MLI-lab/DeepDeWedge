import torch
import tqdm
from utils.missing_wedge import apply_fourier_mask_to_tomo


def masked_loss(model_output, target, rot_mw_mask, mw_mask, mw_weight=2.0):
    joint_filter = rot_mw_mask * mw_mask
    loss = apply_fourier_mask_to_tomo(tomo=target-model_output, mask=joint_filter, output="real").abs().pow(2).mean()
    comp_filter = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
    loss = loss + mw_weight * apply_fourier_mask_to_tomo(tomo=target-model_output, mask=comp_filter, output="real").abs().pow(2).mean()
    return loss


def get_avg_model_input_mean_and_var(loader, batches=None, verbose=False):
    if batches is None:
        batches = 3*len(loader)
    chunks = []
    bar = tqdm.tqdm(range(batches), desc="Getting average model input mean and variance") if verbose else range(batches)
    iter_loader = iter(loader)
    for _ in bar:
        try:
            item = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            item = next(iter_loader)
        vol = apply_fourier_mask_to_tomo(item["model_input"], item["mw_mask"], output="real")
        chunks.append(vol)
    chunks = torch.concat(chunks, 0)
    means = chunks.mean(dim=(-1,-2,-3))
    vars = chunks.var(dim=(-1,-2,-3))
    if verbose:
        print(f"Average model input mean: {means.mean()} (Variance over inputs: {means.var()})")
        print(f"Average model input varariance: {vars.mean()} (Variance over inputs: {vars.var()})")
    return means.mean().cpu().item(), vars.mean().cpu().item()
