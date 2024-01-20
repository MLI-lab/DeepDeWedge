#%%
from src.utils.mrctools import load_mrc_data, save_mrc_data
from src.utils.visualization import plot_tomo_slices
from src.utils.fourier import * 
from matplotlib import pyplot as plt
from src.utils.missing_wedge import get_missing_wedge_mask, apply_fourier_mask_to_tomo
from skimage.filters import gaussian
import torch
from src.utils.visualization import * 

def clamp(vol):
    std = vol.std()
    return vol.clamp(-3*std, 3*std)
# %%
full = load_mrc_data("/media/ssd0/simon/cryo_et_reconstruction/legionella/Legionella.mrc")
even = load_mrc_data("/media/ssd0/simon/cryo_et_reconstruction/legionella/Legionella_even.mrc")
odd = load_mrc_data("/media/ssd0/simon/cryo_et_reconstruction/legionella/Legionella_odd.mrc")

full_crop = torch.nn.functional.avg_pool3d(full[:, 350:1200, 200:].unsqueeze(0), 3, 3).squeeze()
odd_crop = torch.nn.functional.avg_pool3d(odd[:, 350:1200, 200:].unsqueeze(0), 3, 3).squeeze()
even_crop = torch.nn.functional.avg_pool3d(even[:, 350:1200, 200:].unsqueeze(0), 3, 3).squeeze()



save_mrc_data(full_crop, "/media/ssd0/simon/cryo_et_reconstruction/legionella/Legionella_crop_bin3.mrc")
save_mrc_data(even_crop, "/media/ssd0/simon/cryo_et_reconstruction/legionella/Legionella_even_crop_bin3.mrc")
save_mrc_data(odd_crop, "/media/ssd0/simon/cryo_et_reconstruction/legionella/Legionella_odd_crop_bin3.mrc")

# %%
chunk_even = even# [50:350,500:800,200:500]
chunk_even_ft = fft_3d(chunk_even)


mask = get_missing_wedge_mask(grid_size=chunk_even.shape, mw_angle=60)
chunk_even_filt = apply_fourier_mask_to_tomo(chunk_even, mask=mask)

print((chunk_even_filt-chunk_even).norm() / chunk_even.norm())

# %%
id = 3
base = f"/media/ssd0/simon/cryo_et_reconstruction/legionella/tomos/TS_20231110_LegionellaMutV_{id}"
tomo_file = base + ".mrc"
tomo_even_file = base + "_even.mrc"
tomo_odd_file = base + "_odd.mrc"
mask_file = base + "_mask.mrc"

tomo = load_mrc_data(tomo_file)
tomo_even = load_mrc_data(tomo_even_file)
tomo_odd = load_mrc_data(tomo_odd_file)
mask = load_mrc_data(mask_file)

# tomo1
# sl_y = slice(350, -1)
# sl_x = slice(250, 870)

# tomo2
# sl_y = slice(0, 930)
# sl_x = slice(250, 900)

# tomo3
sl_y = slice(370, 1200)
sl_x = slice(170, -1)


mask_crop = mask[:,sl_y,sl_x]
tomo_crop = tomo[:,sl_y,sl_x]
tomo_even_crop = tomo_even[:,sl_y,sl_x]
tomo_odd_crop = tomo_odd[:,sl_y,sl_x]


plt.figure(figsize=(10,20))
img = tomo_crop.sum(1) 
# img = tomo_crop[:, tomo_crop.shape[1]//2, :]
plt.imshow(img.clamp(-3*img.std(), 3*img.std()))

#%%
save_mrc_data(tomo_crop, base+"_crop.mrc")
save_mrc_data(mask_crop, base+"_crop_mask.mrc")
save_mrc_data(tomo_even_crop, base+"_even_crop.mrc")
save_mrc_data(tomo_odd_crop, base+"_odd_crop.mrc")
# %%
tomo_crop = load_mrc_data("/media/ssd0/simon/cryo_et_reconstruction/legionella/tomos/TS_20231110_LegionellaMutV_1_crop.mrc")
plt.figure(figsize=(10,20))
plt.imshow(tomo_crop.sum(0))
# %%
sl_y = slice(230, -1)
sl_x = slice(250, 900)