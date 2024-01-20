# %%
import math
from matplotlib import pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import os
import shutil
import datetime
import shutil

from pytorch_lightning import seed_everything
from torchinfo import summary

from src.masked_loss import masked_loss
from src.normalization import get_avg_model_input_mean_and_var
from src.refine_tomogram import refine_tomogram
from src.setup_dataset import setup_fitting_and_val_dataset
from src.unet import Unet3D
from src.utils.mrctools import load_mrc_data
from src.utils.visualization import plot_tomo_slices
from src.utils.dataloader import MultiEpochsDataLoader
import numpy as np
from src.utils.fourier import * 


seed_everything(42)

def plot_to_tensorboard(writer, fig, tag, step):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose()
    img = np.swapaxes(img, 2, 1)  # otherwise image is transposed

    # Normalize into 0-1 range for TensorBoard(vol). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image(tag, img, step)
    plt.close(fig)



# %%
class LitUnet3D(pl.LightningModule):
    def __init__(self, unet_params, adam_params):
        super().__init__()
        self.unet_params = unet_params
        self.adam_params = adam_params
        self.unet = Unet3D(**self.unet_params)
        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x.unsqueeze(1)).squeeze(1)  # unsqueeze to add channel dimension, squeeze to remove it

    def training_step(self, batch, batch_idx):
        model_output = self(batch["model_input"])
        loss, _ = masked_loss(
            model_output=model_output,
            target=batch["model_target"],
            rot_mw_mask=batch["rot_mw_mask"],
            mw_mask=batch["mw_mask"],
            mw_weight=2.0,
        )
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss


    def validation_step(self, batch, batch_idx):
        model_output = self(batch["model_input"])
        loss, inside_mw_loss = masked_loss(
            model_output=model_output,
            target=batch["model_target"],
            rot_mw_mask=batch["rot_mw_mask"],
            mw_mask=batch["mw_mask"],
        )
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_inside_mw_loss", inside_mw_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_val_plots(batch_idx=batch_idx, vol_row_0=batch["model_input"], vol_row_1=batch["rot_mw_mask"], vol_row_2=model_output, mode="two_wedge")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.adam_params)
        return optimizer

    def log_val_plots(self, batch_idx, vol_row_0, vol_row_1, vol_row_2, mode):
        idx = vol_row_0.shape[0] // 2
        vol_row_0 = vol_row_0[idx].cpu().squeeze()
        vol_row_1 = vol_row_1[idx].cpu().squeeze()
        vol_row_2 = vol_row_2[idx].cpu().squeeze()
        # log real space images
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        half = vol_row_0.shape[-1] // 2
        for a in ax.flatten():
            a.set_axis_off()
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        ax[0, 0].imshow(vol_row_0[half, :, :])
        ax[0, 1].imshow(vol_row_0[:, half, :])
        ax[0, 2].imshow(vol_row_0[:, :, half])          
        ax[1, 0].imshow(vol_row_1[half, :, :])
        ax[1, 1].imshow(vol_row_1[:, half, :])
        ax[1, 2].imshow(vol_row_1[:, :, half])  
        ax[2, 0].imshow(vol_row_2[half, :, :])
        ax[2, 1].imshow(vol_row_2[:, half, :])
        ax[2, 2].imshow(vol_row_2[:, :, half])  
        plot_to_tensorboard(writer=self.logger.experiment, fig=fig, tag=f"RealRecon_BalBatch={batch_idx}_Chunk={idx}/mode={mode}", step=self.current_epoch)
        # self.logger.experiment.add_figure(f"RealRecon_BalBatch={batch_idx}_Chunk={idx}/mode={mode}", fig, self.current_epoch)

        # log k-space images 
        vol_row_0 = fft_3d(vol_row_0).abs().log()
        vol_row_1 = fft_3d(vol_row_1).abs().log()
        vol_row_2 = fft_3d(vol_row_2).abs().log()
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        half = vol_row_0.shape[-1] // 2
        for a in ax.flatten():
            a.set_axis_off()
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        ax[0, 0].imshow(vol_row_0.sum(0)) 
        ax[0, 1].imshow(vol_row_0.sum(1)) 
        ax[0, 2].imshow(vol_row_0.sum(2))          
        ax[1, 0].imshow(vol_row_1.sum(0)) 
        ax[1, 1].imshow(vol_row_1.sum(1)) 
        ax[1, 2].imshow(vol_row_1.sum(2))  
        ax[2, 0].imshow(vol_row_2.sum(0)) 
        ax[2, 1].imshow(vol_row_2.sum(1)) 
        ax[2, 2].imshow(vol_row_2.sum(2))  
        plot_to_tensorboard(writer=self.logger.experiment, fig=fig, tag=f"FourierRecon_ValBatch={batch_idx}_Chunk={idx}/mode={mode}", step=self.current_epoch)
        # self.logger.experiment.add_figure(f"FourierRecon_ValBatch={batch_idx}_Chunk={idx}/mode={mode}", fig, self.current_epoch)

# %%
if __name__ == "__main__":
    st_dir = f"./subtomos_{datetime.datetime.now()}"

    base_files = [
        "/media/ssd0/simon/cryo_et_reconstruction/legionella/tomos/TS_20231110_LegionellaMutV_2",
        "/media/ssd0/simon/cryo_et_reconstruction/legionella/tomos/TS_20231110_LegionellaMutV_3",
        "/media/ssd0/simon/cryo_et_reconstruction/legionella/tomos/TS_20231110_LegionellaMutV_7",
    ]

    fitting_dataset, val_dataset = setup_fitting_and_val_dataset(
        tomo0_files=[f"{base_file}_even_crop_bin3.mrc" for base_file in base_files],
        tomo1_files=[f"{base_file}_odd_crop_bin3.mrc" for base_file in base_files],
        mask_files=[f"{base_file}_crop_mask_bin3.mrc" for base_file in base_files],
        min_signal_fraction=0.2,
        subtomo_size=128,
        subtomo_extraction_strides=[133-128, 32, 32],  # reducing the stride lengths increases the number of subtomos
        mw_angle=60,  # the width of the missing wedge
        val_fraction=0.20,
        save_subtomos_to=st_dir,
        enlarge_subtomos_for_rotating=False,
    )
    item = fitting_dataset[0]
    # fitting_dataset, val_dataset = setup_fitting_and_val_dataset(
    #     tomo0_files=["/workspaces/DeepDeWedge/tutorial_data/tomo_even_frames.mrc"],
    #     tomo1_files=["/workspaces/DeepDeWedge/tutorial_data/tomo_odd_frames.mrc"],
    #     mask_files=["/workspaces/DeepDeWedge/tutorial_data/all_ones_mask.mrc"],
    #     #mask_files=[f"{base_file}_crop_mask.mrc" for base_file in base_files],
    #     min_signal_fraction=0.3,
    #     subtomo_size=96,
    #     subtomo_extraction_strides=[32, 96, 96],  # reducing the stride lengths increases the number of subtomos
    #     mw_angle=60,  # the width of the missing wedge
    #     val_fraction=0.20,
    #     save_subtomos_to=st_dir,
    # )


    print(f"Number of subtomos for model fitting: {len(fitting_dataset)}")
    print(f"Number of subtomos for validation: {len(val_dataset)}")

    # %%
    # for k in range(3):
    #     item = fitting_dataset[k]
    #     model_input = item["model_input"]
    #     model_input -= model_input.mean()
    #     plot_tomo_slices(item["model_input"], domain="image").show()
    #     plot_tomo_slices(item["model_input"], domain="fourier").show()


    # %% [markdown]
    # We create a fitting and a validation dataloader which return batches of elements from the fitting and validation sets:

    # %%
    batch_size = 6  # you may have to reduce the batch size if your GPU runs out of memory
    num_workers = 10
   # more workers enable faster data loading but use more CPU resources

    fitting_dataloader = MultiEpochsDataLoader(
        dataset=fitting_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = MultiEpochsDataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # %%
    unet_params = {
        "in_chans": 1,
        "out_chans": 1,
        "chans": 32,
        "num_downsample_layers": 3,
        "drop_prob": 0.0,
    }

    avg_model_input_mean, avg_model_input_var = get_avg_model_input_mean_and_var(
        fitting_dataloader,
        batches=1*len(fitting_dataloader),  # compute mean and variance of the model inputs using three passes through the entire dataset
        verbose=True,
    )
    unet_params["normalization_loc"] = avg_model_input_mean
    unet_params["normalization_scale"] = math.sqrt(avg_model_input_var)
    # unet_params["normalization_loc"] = 0.000
    # unet_params["normalization_scale"] = math.sqrt(0.66)


    #%%
    lit_unet = LitUnet3D(unet_params=unet_params, adam_params={"lr": 4e-3})
    # setup a logger that logs the metrics (train_loss, val_loss) we specified above
    logger = pl.loggers.TensorBoardLogger("/media/ssd0/simon/cryo_et_recontruction/tensorboard_logs", name="dummy")  # the TensorBoard logger offers more functionality than the CSVLogger,
    logdir = f"{logger.save_dir}/{logger.name}/version_{logger.version}"
    print(f"Saving logs and model checkpoints to '{logdir}'")

    # this saves the model everey 50 epochs
    epoch_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{logdir}/checkpoints/epoch", 
        filename='{epoch}', 
        monitor="epoch", 
        verbose=True,
        save_top_k=-1,
        every_n_epochs=10, 
    )
    # this saves the top 3 models with the lowest validation loss
    val_loss_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{logdir}/checkpoints/val_loss", 
        filename='{epoch}-{val_loss:.5f}', 
        monitor="val_loss", 
        verbose=True,
        save_top_k=3,
    )

    # initialize the trainer
    trainer = pl.Trainer(
        max_epochs=5000,  # fitting for 1000 epochs may take a long time depending on your hardware; 300 epochs should already give you a good resut
        accelerator="gpu",
        devices=[1],
        check_val_every_n_epoch=5,
        deterministic=True,  # to ensure reproducability
        logger=logger,  
        callbacks=[epoch_callback, val_loss_callback],
        # overfit_batches=1,
        # resume_from_checkpoint="/media/ssd0/simon/cryo_et_recontruction/tensorboard_logs/legionella/version_25/checkpoints/val_loss/epoch=973-val_loss=0.61117.ckpt"
        # resume_from_checkpoint="/media/ssd0/simon/cryo_et_recontruction/tensorboard_logs/legionella/bin1/checkpoints/epoch/epoch=559.ckpt"
    )

    trainer.fit(lit_unet, fitting_dataloader, val_dataloader)

    shutil.rmtree(st_dir)
    # %%
    # we load a model that we fitted for 1000 epochs, comment this out if you want to use the model you just trained
    # lit_unet = LitUnet3D.load_from_checkpoint("./tutorial_data/fitted_model.ckpt").to("cuda:0")

    # tomo_ref = refine_tomogram(
    #     tomo=tomo_full,
    #     lightning_model=lit_unet.to(lit_unet.device),
    #     subtomo_size=96,  # this should be the same as the subtomo_size used for the model fitting
    #     subtomo_extraction_strides=[64, 64, 64],  # this can differ from the subtomo_extraction_strides used for the model fitting; reduce the stride lengths if you observe artifacts in the refined tomogram
    #     batch_size=10,
    # )
    # tomo_ref = tomo_ref.cpu()
    # plot_tomo_slices(
    #     tomo_ref.clamp(-3 * tomo_ref.std(), 3 * tomo_ref.std()), figsize=(10, 15)
    # ).show()


