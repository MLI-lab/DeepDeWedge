# %%
import os

import torch
from torch.utils.data import Dataset
from scipy import spatial

from utils.mrctools import load_mrc_data
from utils.rotation import rotate_vol_around_axis
from utils.missing_wedge import (
    get_missing_wedge_mask,
    get_rotated_missing_wedge_mask,
    apply_fourier_mask_to_tomo,
)


BASE_SEED = 888


class SubtomoDataset(Dataset):
    def __init__(
        self,
        subtomo_dir,
        crop_subtomos_to_size,
        mw_angle,
        rotate_subtomos=True,
        deterministic_rotations=False,
    ):
        super().__init__()
        self.subtomo_dir = subtomo_dir
        self.crop_subtomos_to_size = crop_subtomos_to_size
        self.mw_angle = mw_angle
        self.deterministic_rotations = deterministic_rotations
        self.rotate_subtomos = rotate_subtomos

    def _load_subtomo_pair(self, index):
        subtomo0 = load_mrc_data(f"{self.subtomo_dir}/subtomo0/{index}.mrc")
        subtomo1 = load_mrc_data(f"{self.subtomo_dir}/subtomo1/{index}.mrc")
        return subtomo0, subtomo1

    def _sample_rot_axis_and_angle(self, index):
        seed = BASE_SEED + index if self.deterministic_rotations else None
        rotvec = torch.from_numpy(
            spatial.transform.Rotation.random(random_state=seed).as_rotvec()
        )
        rot_axis = rotvec / rotvec.norm()
        rot_angle = torch.rad2deg(rotvec.norm())
        return rot_axis, rot_angle

    def __len__(self):
        return len(os.listdir(f"{self.subtomo_dir}/subtomo0"))

    def __getitem__(self, index):
        subtomo0, subtomo1 = self._load_subtomo_pair(index)
        # rotate subtomos
        if self.rotate_subtomos:
            rot_axis, rot_angle = self._sample_rot_axis_and_angle(index)
        else:
            rot_angle, rot_axis = 0, torch.tensor([1.0, 0.0, 0.0])
        subtomo0 = rotate_vol_around_axis(
            subtomo0,
            rot_angle=rot_angle,
            rot_axis=rot_axis,
            output_shape=3 * [self.crop_subtomos_to_size],
        )
        subtomo1 = rotate_vol_around_axis(
            subtomo1,
            rot_angle=rot_angle,
            rot_axis=rot_axis,
            output_shape=3 * [self.crop_subtomos_to_size],
        )
        # missing wedge masks
        rot_mw_mask = get_rotated_missing_wedge_mask(
            grid_size=3 * [self.crop_subtomos_to_size],
            mw_angle=self.mw_angle,
            rot_axis=rot_axis,
            rot_angle=rot_angle,
            device=subtomo0.device,
        )
        mw_mask = get_missing_wedge_mask(
            grid_size=3 * [self.crop_subtomos_to_size],
            mw_angle=self.mw_angle,
            device=subtomo0.device,
        )
        model_input = apply_fourier_mask_to_tomo(subtomo0, mw_mask)
        item = {
            "model_input": model_input,
            "model_target": subtomo1,
            "mw_mask": mw_mask,
            "rot_mw_mask": rot_mw_mask,
        }
        return item
