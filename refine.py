#%%
from train import LitUnet3D
from src.refine_tomogram import refine_tomogram
from src.utils.mrctools import load_mrc_data
from src.utils.visualization import plot_tomo_slices
from src.utils.missing_wedge import get_missing_wedge_mask, apply_fourier_mask_to_tomo
import torch
from src.utils.fourier import *
from src.utils.data import SubtomoDataset
import scipy.ndimage
from scipy.ndimage import gaussian_filter

from matplotlib import pyplot as plt

def clamp(vol, k=3):
    std = vol.std()
    return vol.clamp(-k*std, k*std)

#%%
# ckpt = "/media/ssd0/simon/cryo_et_recontruction/tensorboard_logs/dummy/version_0/checkpoints/val_loss/epoch=729-val_loss=1.05612.ckpt"
# ckpt = "/media/ssd0/simon/cryo_et_recontruction/tensorboard_logs/dummy/version_1/checkpoints/val_loss/epoch=794-val_loss=0.47228.ckpt"
# ckpt = "/media/ssd0/simon/cryo_et_recontruction/tensorboard_logs/legionella/bin3_lam=2_lr=4e-3_chans=32_tomos=[2,3,7]_boxsize=88/checkpoints/val_loss/epoch=879-val_loss=0.65599.ckpt"
ckpt = "/media/ssd0/simon/cryo_et_recontruction/tensorboard_logs/legionella/bin3_lam=5_chans=16/checkpoints/val_loss/epoch=19110-val_loss=0.59821.ckpt"
lit_unet = LitUnet3D.load_from_checkpoint(ckpt).to("cuda:3")

tomo_file = "/media/hdd3/simon/cryo_et_reconstruction/legionella/single_tomo/Legionella_crop_bin3.mrc"
# tomo_file = "/media/hdd3/simon/cryo_et_reconstruction/legionella/tomos/TS_20231110_LegionellaMutV_2_crop_bin3.mrc"
tomo_full = load_mrc_data(tomo_file)
# tomo_full = load_mrc_data("/workspaces/DeepDeWedge/tutorial_data/tomo_full.mrc")
# tomo_full = load_mrc_data("/media/hdd3/simon/cryo_et_reconstruction/legionella/tomos/TS_20231110_LegionellaMutV_2_crop.mrc")
# tomo_full = torch.nn.functional.avg_pool3d(tomo_full.unsqueeze(0), 3, 3).squeeze()
# tomo_full = tomo_full[150:350,:,:]
# tomo_full = apply_fourier_mask_to_tomo(tomo_full, mask=get_missing_wedge_mask(grid_size=tomo_full.shape, mw_angle=60))

tomo_full -= tomo_full.mean()
tomo_full /= tomo_full.std()

plot_tomo_slices(tomo_full, figsize=(10, 15))
#%%
subtomo_size = 96
subtomo_extraction_stirdes = [64,64,64]
tomo_ref = refine_tomogram(
    tomo=tomo_full.to(lit_unet.device),
    lightning_model=lit_unet,
    subtomo_size=subtomo_size,  # this should be the same as the subtomo_size used for the model fitting
    subtomo_extraction_strides=subtomo_extraction_stirdes,  # this can differ from the subtomo_extraction_strides used for the model fitting; reduce the stride lengths if you observe artifacts in the refined tomogram
    batch_size=2,
)
tomo_ref = tomo_ref.cpu()
plot_tomo_slices(clamp(tomo_ref), figsize=(10, 15)
).show()

tomo_ref -= tomo_ref.mean()
tomo_ref /= tomo_ref.std()

mw_mask = get_missing_wedge_mask(tomo_ref.shape, 60, device=tomo_ref.device)
tomo_ref_mw = apply_fourier_mask_to_tomo(tomo_ref, mw_mask)

tomo_ref_ref = refine_tomogram(
    tomo=tomo_ref_mw.to(lit_unet.device),
    lightning_model=lit_unet,
    subtomo_size=subtomo_size,  # this should be the same as the subtomo_size used for the model fitting
    subtomo_extraction_strides=subtomo_extraction_stirdes,  # this can differ from the subtomo_extraction_strides used for the model fitting; reduce the stride lengths if you observe artifacts in the refined tomogram
    batch_size=2,
)
tomo_ref_ref = tomo_ref_ref.cpu()
plot_tomo_slices(clamp(tomo_ref_ref), figsize=(10, 15)
).show()


#%%
chunk = tomo_ref#[:400,200:600,200:600]
mw_mask = get_missing_wedge_mask(chunk.shape, 60)
chunk_ft = fft_3d(chunk)
mw_ft = fft_3d(apply_fourier_mask_to_tomo(chunk, torch.ones_like(mw_mask) - mw_mask))

plot_tomo_slices(chunk, "fourier")
print(mw_ft.norm() / chunk.norm())

#%%
sl = 140
plt.figure(figsize=(20,10))
plt.imshow(clamp(tomo_full[:,sl,:]))
plt.figure(figsize=(20,10))
plt.imshow(clamp(tomo_ref[:,sl,:], 3))
# %%
ds = SubtomoDataset(
        subtomo_dir="/workspaces/DeepDeWedge/subtomos_2023-12-19 13:20:58.278111/val_subtomos",
        crop_subtomos_to_size=88,
        mw_angle=59,
        rotate_subtomos=False,
        deterministic_rotations=False,
    )
# %%
for k in range(len(ds)):
    with torch.no_grad():
        item = ds[k]
        inp = item["model_input"]
        # inp = item["subtomo0"][14:-14, 14:-14, 14:-14]
        output = lit_unet(inp.to(lit_unet.device).unsqueeze(0)).squeeze().cpu()
        plot_tomo_slices(item["model_input"])
        plot_tomo_slices(output, "image")
        # for _ in range(10):
        #     item = ds[k]
        #     inp = item["subtomo0"][14:-14, 14:-14, 14:-14]
        #     outputs = []
        #     inp = apply_fourier_mask_to_tomo(inp, item["rot_mw_mask"])
        #     output = lit_unet(inp.to(lit_unet.device).unsqueeze(0)).squeeze().cpu()
        #     outputs.append(output)
        # output = torch.stack(outputs).mean(0)
        # plot_tomo_slices(output, "image")
        # plot_tomo_slices(item["rot_mw_mask"])
        #mw_mask = get_missing_wedge_mask(output.shape, 59)
        #mw_ft = fft_3d(apply_fourier_mask_to_tomo(output, torch.ones_like(mw_mask) - mw_mask))

        print(mw_ft.norm() / output.norm())

# %%
