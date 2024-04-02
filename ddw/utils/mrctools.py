import mrcfile
import torch
import os 
import shutil


def load_mrc_data(mrc_file):
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        try:
            data = torch.tensor(mrc.data)
        except TypeError:
            data = torch.tensor(mrc.data.astype(float))
    return data


def save_mrc_data(data, mrc_file, save=False):
    if save:
        if os.path.exists(mrc_file):
            print(f"File '{mrc_file}' already exists! Moving it to '{mrc_file}~'")
            shutil.move(mrc_file, f"{mrc_file}~")
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(data.numpy())
