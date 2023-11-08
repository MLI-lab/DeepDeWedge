# %%
import os
import math

from utils.mrctools import load_mrc_data, save_mrc_data
from utils.subtomos import extract_subtomos, sample_non_overlapping_subtomo_ids
from utils.data import SubtomoDataset


# %%
def setup_fitting_and_val_dataset(
    tomo0_files,
    tomo1_files,
    subtomo_size,
    mw_angle,
    val_fraction,
    subtomo_extraction_strides=None,
    pad_before_subtomo_extraction=False,
    rotate_subtomos=True,
    save_subtomos_to="./subtomos/",
):
    fitting_subtomo_dir = f"{save_subtomos_to}/fitting_subtomos"
    val_subtomo_dir = f"{save_subtomos_to}/val_subtomos"

    os.makedirs(f"{fitting_subtomo_dir}/subtomo0/", exist_ok=False)
    os.makedirs(f"{fitting_subtomo_dir}/subtomo1/", exist_ok=False)
    os.makedirs(f"{val_subtomo_dir}/subtomo0/", exist_ok=False)
    os.makedirs(f"{val_subtomo_dir}/subtomo1/", exist_ok=False)

    fitting_counter, val_counter = 0, 0
    for tomo0_file, tomo1_file in zip(tomo0_files, tomo1_files):
        tomo0 = load_mrc_data(tomo0_file)
        subtomos0, start_coords = extract_subtomos(
            tomo=tomo0,
            subtomo_size=subtomo_size,
            subtomo_extraction_strides=subtomo_extraction_strides,
            enlarge_subtomos_for_rotating=rotate_subtomos,
            pad_before_subtomo_extraction=pad_before_subtomo_extraction,
        )
        tomo1 = load_mrc_data(tomo1_file)
        subtomos1, _ = extract_subtomos(
            tomo=tomo1,
            subtomo_size=subtomo_size,
            subtomo_extraction_strides=subtomo_extraction_strides,
            enlarge_subtomos_for_rotating=rotate_subtomos,
            pad_before_subtomo_extraction=pad_before_subtomo_extraction,
        )
        val_ids = sample_non_overlapping_subtomo_ids(
            subtomo_start_coords=start_coords,
            subtomo_size=subtomo_size,
            n=math.floor(len(subtomos0) * val_fraction),
        )
        fitting_ids = [k for k in range(len(subtomos0)) if k not in val_ids]

        for idx in fitting_ids:
            save_mrc_data(
                subtomos0[idx], f"{fitting_subtomo_dir}/subtomo0/{fitting_counter}.mrc"
            )
            save_mrc_data(
                subtomos1[idx], f"{fitting_subtomo_dir}/subtomo1/{fitting_counter}.mrc"
            )
            fitting_counter += 1

        for idx in val_ids:
            save_mrc_data(
                subtomos0[idx], f"{val_subtomo_dir}/subtomo0/{val_counter}.mrc"
            )
            save_mrc_data(
                subtomos1[idx], f"{val_subtomo_dir}/subtomo1/{val_counter}.mrc"
            )
            val_counter += 1

    fitting_dataset = SubtomoDataset(
        subtomo_dir=fitting_subtomo_dir,
        crop_subtomos_to_size=subtomo_size,
        mw_angle=mw_angle,
        rotate_subtomos=rotate_subtomos,
        deterministic_rotations=False,
    )

    val_dataset = SubtomoDataset(
        subtomo_dir=val_subtomo_dir,
        crop_subtomos_to_size=subtomo_size,
        mw_angle=mw_angle,
        rotate_subtomos=rotate_subtomos,
        deterministic_rotations=True,
    )

    return fitting_dataset, val_dataset
