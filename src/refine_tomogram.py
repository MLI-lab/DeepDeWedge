# %%
import torch
import tqdm
import numpy as np
from .utils.subtomos import extract_subtomos, reassemble_subtomos
from .utils.missing_wedge import get_rotated_missing_wedge_mask
from .utils.fourier import apply_fourier_mask_to_tomo
import scipy.spatial
from torch.utils.data import DataLoader, TensorDataset

BASE_ROTATION_SEED = 777

def clamp(vol, factor=3):
    return vol.clamp(-factor * vol.std(), factor * vol.std())


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, subtomos, add_wedge=False, mw_angle=60, deterministic=False):
        self.subtomos = subtomos
        self.add_wedge = add_wedge
        self.mw_angle = mw_angle
        self.deterministic = deterministic
    
    def __len__(self):
        return len(self.subtomos)

    def _sample_rot_axis_and_angle(self, index):
        if self.deterministic: 
            seed = BASE_ROTATION_SEED+index
        else:
            seed = None
        rotvec = torch.from_numpy(scipy.spatial.transform.Rotation.random(random_state=seed).as_rotvec())
        rot_axis = rotvec / rotvec.norm()
        rot_angle = torch.rad2deg(rotvec.norm())
        return rot_axis, rot_angle

    def __getitem__(self, index):
        subtomo = self.subtomos[index]
        if self.add_wedge:
            axis, angle = self._sample_rot_axis_and_angle(index)
            mw_filter = get_rotated_missing_wedge_mask(grid_size=subtomo.shape, mw_angle=self.mw_angle, rot_axis=axis, rot_angle=angle)
            subtomo = apply_fourier_mask_to_tomo(subtomo, mw_filter)
        return subtomo



def refine_tomogram(
    tomo,
    lightning_model,
    subtomo_size=None,
    subtomo_extraction_strides=None,
    num_repetitions=1,
    add_wedge=False,
    mw_angle=60,
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
    tomo += lightning_model.unet.normalization_loc.to(tomo.device)
    tomo *= lightning_model.unet.normalization_scale.to(tomo.device)

    subtomos, subtomo_start_coords = extract_subtomos(
        tomo=tomo.cpu(),
        subtomo_size=subtomo_size,
        subtomo_extraction_strides=subtomo_extraction_strides,
        enlarge_subtomos_for_rotating=False,
        pad_before_subtomo_extraction=True,    )

    subtomos = InferenceDataset(
        subtomos=torch.stack(subtomos),
        add_wedge=add_wedge,
        mw_angle=mw_angle,
        deterministic=False
    )

    tomo_ref = torch.zeros_like(tomo)
    for _ in range(num_repetitions):
        subtomo_loader = DataLoader(
            subtomos,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            model_outputs = []
            for batch in tqdm.tqdm(subtomo_loader, desc="Processing subtomos"):
                batch_subtomos = batch.to(lightning_model.device)
                model_output = lightning_model(batch_subtomos)
                model_outputs.append(model_output.detach())
            model_outputs = list(torch.concat(model_outputs, 0))

        tomo_ref_ = reassemble_subtomos(
            subtomos=model_outputs,
            subtomo_start_coords=subtomo_start_coords,
            crop_to_size=tomo.shape,
        )
        tomo_ref += tomo_ref_
    return tomo_ref / num_repetitions
