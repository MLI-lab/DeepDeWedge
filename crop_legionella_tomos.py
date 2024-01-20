#%%
from src.utils.mrctools import *
from matplotlib import pyplot as plt

id = 8
base = f"/media/ssd0/simon/cryo_et_reconstruction/legionella/tomos/TS_20231110_LegionellaMutV_{id}"
tomo_file = base + ".mrc"
tomo_even_file = base + "_even.mrc"
tomo_odd_file = base + "_odd.mrc"

tomo = load_mrc_data(tomo_file)
tomo_even = load_mrc_data(tomo_even_file)
tomo_odd = load_mrc_data(tomo_odd_file)

# id = 1
# sl_y = slice(350, -1)
# sl_x = slice(250, 870)

# id = 2
# sl_y = slice(0, 930)
# sl_x = slice(250, 900)

# id = 3
# sl_y = slice(370, 1200)
# sl_x = slice(170, -1)

# id = 5
# sl_y = slice(440, -1)
# sl_x = slice(290, 900)

# id = 6
# sl_y = slice(0, -1)
# sl_x = slice(0, 820)

# id = 7
# sl_y = slice(280, 950)
# sl_x = slice(270, -1)

# id = 8
sl_y = slice(35, 1000)
sl_x = slice(350, -1)

tomo_crop = tomo[:,sl_y,sl_x]
tomo_even_crop = tomo_even[:,sl_y,sl_x]
tomo_odd_crop = tomo_odd[:,sl_y,sl_x]


plt.figure(figsize=(10,20))
plt.imshow(tomo_crop.sum(0))


#%%
save_mrc_data(tomo_crop, base+"_crop.mrc")
save_mrc_data(tomo_even_crop, base+"_even_crop.mrc")
save_mrc_data(tomo_odd_crop, base+"_odd_crop.mrc")
# %%
