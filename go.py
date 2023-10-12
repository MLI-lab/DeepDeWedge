#%%
import torch
import math
from subtomo_dataset import setup_fitting_and_val_dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from utils.misc import load_mrc_data

from unet import Unet3D
from utils.fitting import get_avg_model_input_mean_and_var, masked_loss
from utils.misc import save_mrc_data
from utils.visualization import plot_tomo_slices, plot_vol_fft_slices
import datetime
import shutil



class LightningUnet3D(pl.LightningModule):
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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self(batch["model_input"])
        loss = masked_loss(
            model_output=model_output,
            target=batch["model_target"], 
            rot_mw_mask=batch["rot_mw_mask"], 
            mw_mask=batch["mw_mask"]
        )
        self.logger.experiment.add_figure("model_output", plot_tomo_slices(model_output[0].cpu()), self.global_step)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.adam_params)
        return optimizer

# %%
if __name__ == "__main__":
    seed_everything(42, workers=True)
    #%%
    try:
        shutil.rmtree("./subtomos")
    except:
        print("No subtomos folder to remove")
    dataset_params = {
        "vol_even_files": ["/media/hdd1/simon/cryo_et_reconstruction/Tomo110/recons/frames=even/split=full/ctfcorrection=True/Tomo110_mw60_tight.mrc"],
        "vol_odd_files": ["/media/hdd1/simon/cryo_et_reconstruction/Tomo110/recons/frames=odd/split=full/ctfcorrection=True/Tomo110_mw60_tight.mrc"],
        "subtomo_size": 96,
        "extraction_strides": [179-136, 80, 80],
        "mw_angle": 60,
        "rotate_subtomos": True,
        "val_fraction": 0.2
    }
    fitting_dataset, val_dataset = setup_fitting_and_val_dataset(**dataset_params)

    #%%
    dataloader_params = {
        "batch_size": 5,
        "num_workers": 10,
    } 
    fitting_dataloader = MultiEpochsDataLoader(dataset=fitting_dataset, **dataloader_params, pin_memory=True)
    val_dataloader = MultiEpochsDataLoader(dataset=val_dataset, **dataloader_params, shuffle=False, pin_memory=True)

    #%%
    unet_params = {
        "chans": 64,
        "num_pool_layers": 3,
    }

    adam_params = {
        "lr": 4e-4,
    }
    avg_model_input_mean, avg_model_input_var = get_avg_model_input_mean_and_var(fitting_dataloader,verbose=True)
    # avg_model_input_mean, avg_model_input_var = 0.0, 2.0
    unet_params["normalization_loc"] = avg_model_input_mean
    unet_params["normalization_scale"] = math.sqrt(avg_model_input_var)

    lu = LightningUnet3D(
        unet_params=unet_params,
        adam_params=adam_params,
    )

    #%%
    trainer_params = {
        "max_epochs": 1000,
        "accelerator": "gpu",
        "devices": [3],
        "log_every_n_steps": 1,
        "check_val_every_n_epoch": 10,
        "num_sanity_val_steps": 2,
        "logger": pl.loggers.TensorBoardLogger("tensorboard_logs", name="tomo110"),
        "resume_from_checkpoint": "/workspaces/cryo_et_github/tensorboard_logs/tomo110/version_0/checkpoints/epoch=129-step=4290.ckpt",
    }
    trainer = pl.Trainer(**trainer_params)

    #%%
    trainer.fit(lu, fitting_dataloader, val_dataloader)
    print(f"End training: {datetime.datetime.now()}")

# %%
