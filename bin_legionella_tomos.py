#%%
from src.utils.mrctools import *
from matplotlib import pyplot as plt
import torch

bin = 3

for id in [2, 3, 7]:
    base = f"/media/ssd0/simon/cryo_et_reconstruction/legionella/tomos/TS_20231110_LegionellaMutV_{id}"
    for suffix in ["_crop.mrc", "_even_crop.mrc", "_odd_crop.mrc", "_crop_mask.mrc"]:
        vol = load_mrc_data(base + suffix)
        vol_bin = torch.nn.functional.avg_pool3d(vol.unsqueeze(0), bin, bin).squeeze()
        save_mrc_data(vol_bin, base + suffix.replace(".mrc", f"_bin{bin}.mrc"))
