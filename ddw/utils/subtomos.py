# %%
import math
import numpy as np
import torch
import random


def extract_subtomos(
    tomo,
    subtomo_size,
    subtomo_extraction_strides=None,
    enlarge_subtomos_for_rotating=False,
    pad_before_subtomo_extraction=False,
):
    # TODO: refactor subtomo_extraction_strides to subtomo_overlap
    if enlarge_subtomos_for_rotating:
        subtomo_size = ceil_to_even_integer(math.sqrt(2) * subtomo_size)
    if subtomo_extraction_strides is None:
        subtomo_extraction_strides = 3 * [subtomo_size]
    if pad_before_subtomo_extraction:
        # pad for subtomo extraction with extraction strides
        pad_x = subtomo_extraction_strides[0] - (
            (tomo.shape[0] - subtomo_size) % subtomo_extraction_strides[0]
        )
        pad_y = subtomo_extraction_strides[1] - (
            (tomo.shape[1] - subtomo_size) % subtomo_extraction_strides[1]
        )
        pad_z = subtomo_extraction_strides[2] - (
            (tomo.shape[2] - subtomo_size) % subtomo_extraction_strides[2]
        )
        # pad = torch.nn.ConstantPad3d((0, pad_z, 0, pad_y, 0, pad_x), 0)  # right pad with zero
        pad = torch.nn.ReflectionPad3d((0, pad_z, 0, pad_y, 0, pad_x))
        tomo = pad(tomo.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    # Generating starting indices for each subtomo
    subtomo_start_coords = [
        (i, j, k)
        for i in range(
            0, tomo.shape[0] - subtomo_size + 1, subtomo_extraction_strides[0]
        )
        for j in range(
            0, tomo.shape[1] - subtomo_size + 1, subtomo_extraction_strides[1]
        )
        for k in range(
            0, tomo.shape[2] - subtomo_size + 1, subtomo_extraction_strides[2]
        )
    ]
    subtomos = (
        tomo.unfold(0, subtomo_size, subtomo_extraction_strides[0])
        .unfold(1, subtomo_size, subtomo_extraction_strides[1])
        .unfold(2, subtomo_size, subtomo_extraction_strides[2])
    )
    subtomos = subtomos.reshape(-1, subtomo_size, subtomo_size, subtomo_size)
    subtomos = list(subtomos)
    return subtomos, subtomo_start_coords


def get_linear_ramp_weights(subtomo_size, subtomo_overlap):
    ramp = np.linspace(0, 1, subtomo_overlap) + 1e-6
    weight_map_1d = np.ones(subtomo_size)
    weight_map_1d[:subtomo_overlap] = ramp  # Apply sigmoid ramp at the start
    weight_map_1d[-subtomo_overlap:] = ramp[::-1]  # and at the end, inverted
    
    # Create a 3D weight map by extending the 1D weight map to 3 dimensions
    weight_map_3d = np.ones((subtomo_size, subtomo_size, subtomo_size))
    for i in range(subtomo_size):
        for j in range(subtomo_size):
            for k in range(subtomo_size):
                weight_map_3d[i, j, k] = weight_map_1d[i] * weight_map_1d[j] * weight_map_1d[k]
                
    return torch.from_numpy(weight_map_3d)

def reassemble_subtomos(subtomos, subtomo_start_coords, subtomo_overlap=None, crop_to_size=None):
    # calculate the max indices in each dimension to infer the shape of the original tomogram    
    subtomo_size = subtomos[0].shape[0]
    max_idx = [
        max(start_idx[i] + subtomo_size for start_idx in subtomo_start_coords)
        for i in range(3)
    ]
    if subtomo_overlap is None:
        subtomo_weights = torch.ones_like(subtomos[0])
    else:
        subtomo_weights = get_linear_ramp_weights(subtomos[0].shape[0], subtomo_overlap).to(subtomos[0].device)

    out_vol = torch.zeros(max_idx, dtype=torch.float32, device=subtomos[0].device)
    count_vol = torch.zeros_like(out_vol)
    for subtomo, start_idx in zip(subtomos, subtomo_start_coords):
        end_idx = [start + subtomo_size for start in start_idx]
        out_vol[
            start_idx[0] : end_idx[0],
            start_idx[1] : end_idx[1],
            start_idx[2] : end_idx[2],
        ] += subtomo * subtomo_weights
        count_vol[
            start_idx[0] : end_idx[0],
            start_idx[1] : end_idx[1],
            start_idx[2] : end_idx[2],
        ] += subtomo_weights
    # avoid division by zero by replacing zero counts with ones
    # count_vol[count_vol == 0] = 1
    # average the overlapping regions by dividing the accumulated values by their count
    out_vol /= count_vol
    if crop_to_size is not None:
        out_vol = out_vol[: crop_to_size[0], : crop_to_size[1], : crop_to_size[2]]
    return out_vol


def try_to_sample_non_overlapping_subtomo_ids(
    subtomo_start_coords, subtomo_size, target_sample_size, max_tries=1, verbose=True
):
    n = 0
    most_non_overlapping_subtomo_ids = []
    while n < max_tries:
        non_overlapping_subtomo_ids = try_to_sample_non_overlapping_subtomo_ids_(
            subtomo_start_coords, subtomo_size, target_sample_size
        )
        if len(non_overlapping_subtomo_ids) == target_sample_size:
            return non_overlapping_subtomo_ids
        elif len(non_overlapping_subtomo_ids) > len(most_non_overlapping_subtomo_ids):
            most_non_overlapping_subtomo_ids = non_overlapping_subtomo_ids
            n += 1
    if verbose:
        print(
            f"Warning: Could not sample {target_sample_size} non-overlapping subtomos. "
        )
    return most_non_overlapping_subtomo_ids


# this was written with the help of chatgpt and copoilot
def try_to_sample_non_overlapping_subtomo_ids_(
    subtomo_start_coords, subtomo_size, target_sample_size
):
    if target_sample_size > len(subtomo_start_coords):
        raise ValueError("n should be less than or equal to the number of subtomos")

    candidate_ids = list(range(len(subtomo_start_coords)))
    non_overlapping_subtomo_ids = []

    n_rejected = 0
    while len(non_overlapping_subtomo_ids) < target_sample_size:
        if len(candidate_ids) == 0:
            return non_overlapping_subtomo_ids
        idx = random.choice(candidate_ids)
        starting_index = subtomo_start_coords[idx]
        # check if sampled subtomogram overlaps with any of the already selected subtomograms
        overlap = any(
            [
                check_subtomo_overlap(
                    starting_index, subtomo_start_coords[idx], subtomo_size
                )
                for idx in non_overlapping_subtomo_ids
            ]
        )
        if not overlap:
            non_overlapping_subtomo_ids.append(idx)
        else:
            n_rejected += 1
        # remove the sampled subtomogram from the list of indices to sample from
        candidate_ids.remove(idx)
    return non_overlapping_subtomo_ids


def check_subtomo_overlap(starting_index1, starting_index2, subtomo_size):
    # get cube vertices
    vertices1 = get_cube_vertices(starting_index1, subtomo_size)
    vertices2 = get_cube_vertices(starting_index2, subtomo_size)
    intersect = check_cube_overlap(vertices1, vertices2)
    return intersect


def get_cube_vertices(starting_point, cube_size):
    # get cube vertices
    vertices = []
    for k in range(3):
        for j in range(2):
            for i in range(2):
                vertex = list(starting_point)
                vertex[k] += cube_size * i
                vertex[(k + 1) % 3] += cube_size * j
                vertices.append(vertex)
    vertices = torch.tensor(vertices)
    return vertices


def check_cube_overlap(vertices1, vertices2):
    intersect_x = (vertices1.min(0).values[0] < vertices2.max(0).values[0]).all() and (
        vertices1.max(0).values[0] > vertices2.min(0).values[0]
    ).all()
    intersect_y = (vertices1.min(0).values[1] < vertices2.max(0).values[1]).all() and (
        vertices1.max(0).values[1] > vertices2.min(0).values[1]
    ).all()
    intersect_z = (vertices1.min(0).values[2] < vertices2.max(0).values[2]).all() and (
        vertices1.max(0).values[2] > vertices2.min(0).values[2]
    ).all()
    intersect = intersect_x and intersect_y and intersect_z
    return intersect


def ceil_to_even_integer(x):
    return int(math.ceil(x / 2.0) * 2)


# %%
