# %%
import torch
import tqdm
import numpy as np
from .utils.subtomos import extract_subtomos, reassemble_subtomos
from torch.utils.data import DataLoader, TensorDataset


def clamp(vol, factor=3):
    return vol.clamp(-factor * vol.std(), factor * vol.std())


def refine_tomogram(
    tomo,
    lightning_model,
    subtomo_size=None,
    subtomo_extraction_strides=None,
    batch_size=None,
):
    if subtomo_size is None:
        subtomo_size = lightning_model.hparams.dataset_params["subtomo_size"]
    if subtomo_extraction_strides is None:
        subtomo_extraction_strides = 3 * [subtomo_size]
    if batch_size is None:
        batch_size = lightning_model.hparams.dataloader_params["batch_size"]

    tomo -= tomo.mean()
    tomo /= tomo.std()
    tomo *= lightning_model.unet.normalization_scale.to(tomo.device)
    tomo += lightning_model.unet.normalization_loc.to(tomo.device)

    subtomos, subtomo_start_coords = extract_subtomos(
        tomo=tomo.cpu(),
        subtomo_size=subtomo_size,
        subtomo_extraction_strides=subtomo_extraction_strides,
        enlarge_subtomos_for_rotating=False,
        pad_before_subtomo_extraction=True,
    )

    subtomos = TensorDataset(torch.stack(subtomos))
    subtomo_loader = DataLoader(
        subtomos,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        model_outputs = []
        for batch in tqdm.tqdm(subtomo_loader, desc="Processing subtomos"):
            batch_subtomos = batch[0].to(lightning_model.device)
            model_output = lightning_model(batch_subtomos)
            model_outputs.append(model_output.detach())
        model_outputs = list(torch.concat(model_outputs, 0))

    refined_vol = reassemble_subtomos(
        subtomos=model_outputs,
        subtomo_start_coords=subtomo_start_coords,
        crop_to_size=tomo.shape,
    )
    return refined_vol
