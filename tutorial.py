# %% [markdown]
# # DeepDeWedge Tutorial
# 
# This is a minimal example for how to apply DeepDeWedge. We apply DeepDeWedge to the Wiener-Filter CTF corrected FBP reconstructon of tilt series 05 of EMPIAR-10045.

# %%
import torch
import math
from subtomo_dataset import setup_fitting_and_val_dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from utils.misc import load_mrc_data
from matplotlib import pyplot as plt

from unet import Unet3D
from utils.fitting import get_avg_model_input_mean_and_var, masked_loss
from utils.misc import save_mrc_data
from utils.visualization import plot_tomo_slices
import datetime
import shutil
from utils.filters import fft_3d
from utils.dataloader import MultiEpochsDataLoader
from torchsummary import summary

class LitUnet3D(pl.LightningModule):
    def __init__(self, unet_params, adam_params):
        super().__init__()
        self.unet_params = unet_params
        self.adam_params = adam_params
        self.unet = Unet3D(**unet_params)
        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x.unsqueeze(1)).squeeze(1)  # unsqueeze to add channel dimension, squeeze to remove it

    def training_step(self, batch, batch_idx):
        model_output = self(batch["model_input"])  
        loss = masked_loss(
            model_output=model_output, 
            target=batch["model_target"], 
            rot_mw_mask=batch["rot_mw_mask"], 
            mw_mask=batch["mw_mask"]
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self(batch["model_input"])
        loss = masked_loss(
            model_output=model_output,
            target=batch["model_target"], 
            rot_mw_mask=batch["rot_mw_mask"], 
            mw_mask=batch["mw_mask"]
        )
        self.logger.experiment.add_figure("model_output", plot_tomo_slices(model_output[0].cpu(), return_figure=True), self.global_step)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.adam_params)
        return optimizer



# %% [markdown]
# ## Dataset for Model Fitting
# We first load the dataset for model fitting. The function `setup_fitting_and_val_dataset` returns two torch datasets, one for model fitting and one for validation to prevent overfitting. The tomograms corresponding to the filepaths in `tomo0_files` are used to construct the model inputs, while the ones in `tomo1_files` are used to construct the targets. 
# 
# Both the fitting and the validation dataset return model inputs and targets with shape `subtomo_size x subtomo_size x subtomo_size`. These subtomograms are extracted from the tomograms using x, y and z direction strides specified in `extraction_strides`. To reduce RAM consumption during model fitting, all subtomograms are saved to the directory `save_subtomos_to`, which is created if it doe note exist.
# 
# The number of elements in the validation set is at most `validation_frac` times the number of total extracted subtomograms. It may also contain fewer subtomograms since we randomly sample the validation subtomograms such that they have no overlap with the ones used for model fitting. If the sampling procedure was unable to sample enough validation subtomograms, the function prints a warning.

# %%
if __name__ == "__main__":
    shutil.rmtree("./subtomos/", ignore_errors=True)

    fitting_dataset, val_dataset = setup_fitting_and_val_dataset(
        tomo0_files=["/media/ssd0/simon/cryo_et_reconstruction/empiar/10045/data/ribosomes/AnticipatedResults/Tomograms/05/even_bin6_deconv.mrc"],
        tomo1_files=["/media/ssd0/simon/cryo_et_reconstruction/empiar/10045/data/ribosomes/AnticipatedResults/Tomograms/05/odd_bin6_deconv.mrc"],
        subtomo_size=80,
        extraction_strides=[40, 40, 40],
        mw_angle=60,
        val_fraction=0.2,
        save_subtomos_to="./subtomos/",
    )

    print(f"Number of subtomograms for model fitting: {len(fitting_dataset)}")
    print(f"Number of subtomos for validation: {len(val_dataset)}")

    # %% [markdown]
    # Both the fitting and the validation dataset return dictionries containing the following items:
    # * `model_input`: A model input $\tilde{\mathbf{v}}_\varphi^0$ with two missing wedges
    # * `model_target`: A model target $\tilde{\mathbf{v}}_\varphi^1$ with only one missing wedge
    # * `rot_mw_mask`: The rotated missing wedge mask $\mathbf{M}_\varphi$
    # * `mw_mask`: The original missing wedge mask $\mathbf{M}$. This mask is the same for all elements in the dataset.
    # 
    # **Note**: The rotation angles $\varphi$ in the training set are always random, and are re-sampled every time an item is queried. For the validation dataset, we only sample random rotation angles once and every item always has its fixed rotation.
    # 
    # 
    # Let's now have a look at the real and Fourier domain representation of some of the model inputs:

    # %%
    for k in range(5):
        item = fitting_dataset[k]
        model_input = item["model_input"]
        model_input -= model_input.mean()  
        plot_tomo_slices(item["model_input"])

    # %% [markdown]
    # We create a fitting and a validation dataloader which return batches of elements from the fitting and validation sets:

    # %%
    fitting_dataloader = MultiEpochsDataLoader(dataset=fitting_dataset, batch_size=5, num_workers=20, pin_memory=True)
    val_dataloader = MultiEpochsDataLoader(dataset=val_dataset, batch_size=5, num_workers=5, shuffle=False, pin_memory=True)

    # %% [markdown]
    # ## The 3D U-Net
    # 
    # Here, we setup a 3D U-Net, which we then fit. The architecture we implemented in `unet.py`is the same used in the official IsoNet implementation.
    # Before we can start model fitting, we calculate the average mean and variance of the model inputs in the fitting dataset. We use these values to normalize the input to the U-Net during model fitting.

    # %%
    unet_params = {
        "in_chans": 1,
        "out_chans": 1,
        "chans": 64,
        "num_pool_layers": 3,
        "drop_prob": 0.0,
    }

    avg_model_input_mean, avg_model_input_var = get_avg_model_input_mean_and_var(
        fitting_dataloader, 
        batches=3*len(fitting_dataloader),
        verbose=True
    )
    # avg_model_input_mean, avg_model_input_var = 0.0, 2.0
    unet_params["normalization_loc"] = avg_model_input_mean
    unet_params["normalization_scale"] = math.sqrt(avg_model_input_var)

    unet = Unet3D(**unet_params)
    summary(unet, (1, 80, 80, 80))

    # %% [markdown]
    # # Unet as PyTorchLightning Object for Easy Model Fitting
    # For model fitting, we use the PyTorch lightning framework for convenience. Below, we define the class `LitUnet3D`. The class takes parameters for a U-Net and the `torch.optim.Adam` optimizer as input and can then be used to fit the U-Net. The important methods of this class are:
    # * `training_step`: This step handles passing the model inputs provided by the fitting dataloader through the model and calculating the loss. 
    # + `validation_step`: In this step, we can implement any validation routine we like. For simplicity, we just calculate the loss on the validation set to monitor overfitting. Depending on which logger we use, we can also log plots of the model output.

    # %%

    # %%
    unet = LitUnet3D(unet_params=unet_params, adam_params={"lr": 4e-4})

    # %%
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="gpu",
        devices=[1],
        check_val_every_n_epoch=10,
        logger=pl.loggers.TensorBoardLogger("tensorboard_logs", name="empiar"),
    )

    trainer.fit(unet, fitting_dataloader, val_dataloader)

    # %%


    # %%



