# %%
import os
import math

from .utils.mrctools import load_mrc_data, save_mrc_data
from .utils.subtomos import extract_subtomos, try_to_sample_non_overlapping_subtomo_ids
from .utils.data import SubtomoDataset


# %%
def setup_fitting_and_val_dataset(
    tomo0_files,
    tomo1_files,
    mask_files,
    subtomo_size,
    mw_angle,
    val_fraction,
    min_signal_fraction=0,
    subtomo_extraction_strides=None,
    pad_before_subtomo_extraction=False,
    rotate_subtomos=True,
    enlarge_subtomos_for_rotating=True,
    save_subtomos_to="./subtomos/",
):
    fitting_subtomo_dir = f"{save_subtomos_to}/fitting_subtomos"
    val_subtomo_dir = f"{save_subtomos_to}/val_subtomos"

    os.makedirs(f"{fitting_subtomo_dir}/subtomo0/", exist_ok=False)
    os.makedirs(f"{fitting_subtomo_dir}/subtomo1/", exist_ok=False)
    os.makedirs(f"{val_subtomo_dir}/subtomo0/", exist_ok=False)
    os.makedirs(f"{val_subtomo_dir}/subtomo1/", exist_ok=False)

    fitting_counter, val_counter = 0, 0
    for tomo0_file, tomo1_file, mask_file in zip(tomo0_files, tomo1_files, mask_files):
        tomo0 = load_mrc_data(tomo0_file)
        tomo0 -= tomo0.mean()
        tomo0 /= tomo0.std()
        subtomos0, start_coords = extract_subtomos(
            tomo=tomo0,
            subtomo_size=subtomo_size,
            subtomo_extraction_strides=subtomo_extraction_strides,
            enlarge_subtomos_for_rotating=enlarge_subtomos_for_rotating,
            pad_before_subtomo_extraction=pad_before_subtomo_extraction,
        )
        tomo1 = load_mrc_data(tomo1_file)
        tomo1 -= tomo1.mean()
        tomo1 /= tomo1.std()
        subtomos1, _ = extract_subtomos(
            tomo=tomo1,
            subtomo_size=subtomo_size,
            subtomo_extraction_strides=subtomo_extraction_strides,
            enlarge_subtomos_for_rotating=enlarge_subtomos_for_rotating,
            pad_before_subtomo_extraction=pad_before_subtomo_extraction,
        )
        mask = load_mrc_data(mask_file)
        subtomos_mask, _ = extract_subtomos(
            tomo=mask,
            subtomo_size=subtomo_size,
            subtomo_extraction_strides=subtomo_extraction_strides,
            enlarge_subtomos_for_rotating=enlarge_subtomos_for_rotating,
            pad_before_subtomo_extraction=pad_before_subtomo_extraction,
        )
        selected_subtomo_ids = [k for k, submask in enumerate(subtomos_mask) if (submask.sum() / submask.numel()) >= min_signal_fraction]
        print(f"Selected {len(selected_subtomo_ids)}/{len(subtomos0)} subtomos")
        subtomos0 = [subtomo for k, subtomo in enumerate(subtomos0) if k in selected_subtomo_ids]
        subtomos1 = [subtomo for k, subtomo in enumerate(subtomos1) if k in selected_subtomo_ids]
        start_coords = [coords for k, coords in enumerate(start_coords) if k in selected_subtomo_ids]

        val_ids = try_to_sample_non_overlapping_subtomo_ids(
            subtomo_start_coords=start_coords,
            subtomo_size=subtomo_size,
            target_sample_size=math.floor(len(subtomos0) * val_fraction),
            max_tries=3,
            verbose=False,
        )
        if len(val_ids) < math.floor(len(subtomos0) * val_fraction):
            print(
                f"WARNING: Could not sample {round(val_fraction*100, 2)}% of all subtomos for validation due to overlap with the fitting data. "
                f"Continuing with {round((len(val_ids)/len(subtomos0))*100, 2)}% (i.e. {len(val_ids)}) validation subtomos."
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
