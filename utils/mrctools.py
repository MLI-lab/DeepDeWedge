import mrcfile
import torch


def load_mrc_data(mrc_file):
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        try:
            data = torch.from_numpy(mrc.data)
        except TypeError:
            data = torch.from_numpy(mrc.data.astype(float))
    return data


def save_mrc_data(data, mrc_file):
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(data.numpy())
