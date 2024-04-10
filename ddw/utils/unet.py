import math

import pytorch_lightning as pl
import torch
import tqdm
import yaml
from torch import nn

from .fourier import apply_fourier_mask_to_tomo
from .masked_loss import masked_loss
from .mrctools import save_mrc_data
from .normalization import get_avg_model_input_mean_and_std_from_dataloader


class LitUnet3D(pl.LightningModule):
    """
    PyTrochLightning 'wrapper' of a 3D U-Net. This class implements steps for model fitting, validation and logging. This class is the heart of the 'ddw fit-model' command.
    """

    def __init__(
        self,
        unet_params,
        adam_params,
        subtomo_dir,
        update_subtomo_missing_wedges_every_n_epochs=10,
    ):
        super().__init__()
        self.unet_params = unet_params
        self.adam_params = adam_params
        self.subtomo_dir = subtomo_dir
        self.update_subtomo_missing_wedges_every_n_epochs = (
            update_subtomo_missing_wedges_every_n_epochs
        )
        self.unet = Unet3D(**self.unet_params)
        # self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.995)
        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x.unsqueeze(1)).squeeze(
            1
        )  # unsqueeze to add channel dimension, squeeze to remove it

    def training_step(self, batch, batch_idx):
        model_output = self(batch["model_input"])
        loss = masked_loss(
            model_output=model_output,
            target=batch["model_target"],
            rot_mw_mask=batch["rot_mw_mask"],
            mw_mask=batch["mw_mask"],
        )
        self.log(
            "fitting_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self(batch["model_input"])
        loss = masked_loss(
            model_output=model_output,
            target=batch["model_target"],
            rot_mw_mask=batch["rot_mw_mask"],
            mw_mask=batch["mw_mask"],
        )
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    # def on_before_zero_grad(self, optimizer) -> None:
    #     self.ema.update()

    def on_train_start(self) -> None:
        if self.current_epoch == 0:
            self.update_normalization()

    def on_train_epoch_end(self) -> None:
        if (
            self.current_epoch + 1
        ) % self.update_subtomo_missing_wedges_every_n_epochs == 0:  # +1 because the epoch indexing starts at 0
            self.update_subtomo_missing_wedges()
            self.update_normalization()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.adam_params)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return [optimizer]  # , [scheduler]

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
    #     if scheduler is not None:
    #         scheduler.step()

    def update_subtomo_missing_wedges(self):
        """
        Update the missing wedges of model input subtomos.
        """
        # we don't want to rotate the subtomos when updating them, so we create new dataloader objects with rotate_subtomos=False
        datasets = []
        train_loader = self.trainer.train_dataloader.loaders
        train_set = train_loader.dataset
        train_set.rotate_subtomos = False
        datasets.append(train_set)
        # val_dataloaders may be None
        if self.trainer.val_dataloaders is not None:
            val_loader = self.trainer.val_dataloaders[0]
            val_set = val_loader.dataset
            val_set.rotate_subtomos = False
            datasets.append(val_set)
        dataset = torch.utils.data.ConcatDataset(datasets)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers,
        )
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc="Updating subtomo missing wedges"):
                assert batch["rot_angle"].float().norm() == 0
                subtomo_batch = batch["model_input"]
                subtomo_batch = subtomo_batch.to(self.device)
                # subtomo size has to be divisible by 2**num_downsample_layers due to U-Net architecture -> ensure this by padding
                subtomo_dim = subtomo_batch.shape[-1]
                factor = 2 ** self.unet_params["num_downsample_layers"]
                padding = factor * math.ceil(subtomo_dim / factor) - subtomo_dim
                subtomo_batch = torch.nn.functional.pad(
                    subtomo_batch,
                    pad=(0, padding, 0, padding, 0, padding),
                    mode="constant",
                    value=0,
                )
                # forward pass
                subtomo_batch_ref = self.forward(subtomo_batch)
                # update missing wedges
                mw_mask = batch["mw_mask"].to(subtomo_batch.device)
                subtomo_batch = apply_fourier_mask_to_tomo(
                    subtomo_batch, mw_mask
                ) + apply_fourier_mask_to_tomo(subtomo_batch_ref, 1 - mw_mask)
                # remove padding
                subtomo_batch = subtomo_batch[
                    ..., :subtomo_dim, :subtomo_dim, :subtomo_dim
                ]
                for subtomo, file in zip(subtomo_batch, batch["subtomo0_file"]):
                    save_mrc_data(subtomo.cpu(), file)
        train_set.rotate_subtomos = True
        if self.trainer.val_dataloaders is not None:
            val_set.rotate_subtomos = True

    def update_normalization(self):
        """
        Updates the average model input mean and standard deviation used to normalize the sub-tomograms.
        """
        loc, scale = get_avg_model_input_mean_and_std_from_dataloader(
            dataloader=self.trainer.train_dataloader, verbose=True
        )

        # update normalization in unet
        self.unet.normalization_loc = loc
        self.unet.normalization_scale = scale
        # update normalization in hparams
        self.unet_params["normalization_loc"] = loc
        self.unet_params["normalization_scale"] = scale
        self.update_hparam("unet_params", self.unet_params)
        self.log("normalization/loc", loc)
        self.log("normalization/scale", scale)

    def update_hparam(self, hparam, value):
        """
        Update a hyperparameter in the hparams.yaml file.
        """
        logger = self.trainer.logger
        logdir = f"{logger.save_dir}/{logger.name}/version_{logger.version}"
        hparams_file = f"{logdir}/hparams.yaml"
        hparams = yaml.safe_load(open(hparams_file, "r"))
        hparams[hparam] = value
        with open(hparams_file, "w") as f:
            yaml.dump(hparams, f)


class Unet3D(torch.nn.Module):
    """
    PyTorch implementation of a 3D U-Net, which was inspired by the one used in the IsoNet software package (https://github.com/IsoNet-cryoET/IsoNet/tree/master/models/unet)
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        chans: int = 32,
        num_downsample_layers: int = 3,
        drop_prob: float = 0.0,
        residual: bool = True,
        normalization_loc: float = 0.0,
        normalization_scale: float = 1.0,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_downsample_layers = num_downsample_layers
        self.drop_prob = drop_prob
        self.residual = residual
        self.normalization_loc = normalization_loc
        self.normalization_scale = normalization_scale
        self.__init_layers__()

    @property
    def normalization_loc(self):
        return self._normalization_loc

    @normalization_loc.setter
    def normalization_loc(self, normalization_loc):
        self._normalization_loc = nn.parameter.Parameter(
            torch.tensor(normalization_loc), requires_grad=False
        )

    @property
    def normalization_scale(self):
        return self._normalization_scale

    @normalization_scale.setter
    def normalization_scale(self, normalization_scale):
        self._normalization_scale = nn.parameter.Parameter(
            torch.tensor(normalization_scale), requires_grad=False
        )

    def __init_layers__(self):
        self.down_blocks = nn.ModuleList(
            [DownConvBlock(self.in_chans, self.chans, self.drop_prob)]
        )
        self.down_samplers = nn.ModuleList([SpatialDownSampling(self.chans)])

        ch = self.chans
        for _ in range(self.num_downsample_layers - 1):
            self.down_blocks.append(DownConvBlock(ch, ch * 2, self.drop_prob))
            self.down_samplers.append(SpatialDownSampling(ch * 2))
            ch *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch, ch * 2, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(ch * 2, ch, kernel_size=(3, 3, 3), padding=1),
        )

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList([SpatialUpSampling(in_chans=ch, out_chans=ch)])

        for _ in range(self.num_downsample_layers - 1):
            self.up_blocks.append(UpConvBlock(2 * ch, ch, self.drop_prob))
            self.upsamplers.append(SpatialUpSampling(in_chans=ch, out_chans=ch // 2))
            ch //= 2
        self.up_blocks.append(UpConvBlock(2 * ch, ch, self.drop_prob))

        self.final_conv = nn.Conv3d(
            ch, self.out_chans, kernel_size=(1, 1, 1), stride=(1, 1, 1)
        )

    def normalize(self, volume: torch.Tensor) -> torch.Tensor:
        return (volume - self.normalization_loc) / (self.normalization_scale + 1e-6)

    def denormalize(self, volume: torch.Tensor) -> torch.Tensor:
        return volume * (self.normalization_scale + 1e-6) + self.normalization_loc

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        volume = self.normalize(volume)

        stack = []
        output = volume

        # apply down-sampling layers
        for block, downsampler in zip(self.down_blocks, self.down_samplers):
            output = block(output)
            stack.append(output)  # save intermediate outputs for skip connections
            output = downsampler(output)

        output = self.bottleneck(output)

        # apply up-sampling layers
        for upsampler, block in zip(self.upsamplers, self.up_blocks):
            output = upsampler(output, cat=stack.pop())
            output = block(output)

        output = self.final_conv(output)
        if self.residual:
            output = output + volume

        output = self.denormalize(output)
        return output


class DownConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=(3, 3, 3), padding=1),
            nn.InstanceNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=(3, 3, 3), padding=1),
            nn.InstanceNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=(3, 3, 3), padding=1),
            nn.InstanceNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.layers(volume)


class UpConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, in_chans // 2, kernel_size=(3, 3, 3), padding=1),
            nn.InstanceNorm3d(in_chans // 2),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(in_chans // 2, in_chans // 2, kernel_size=(3, 3, 3), padding=1),
            nn.InstanceNorm3d(in_chans // 2),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(in_chans // 2, out_chans, kernel_size=(3, 3, 3), padding=1),
            nn.InstanceNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.layers(volume)


class SpatialDownSampling(nn.Module):
    def __init__(self, chans: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(chans, chans, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume):
        return self.layers(volume)


class SpatialUpSampling(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob=0.0):
        super().__init__()
        self.tconv = nn.ConvTranspose3d(
            in_chans,
            out_chans,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=1,
            output_padding=1,
        )
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)

    def forward(self, volume: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        output = self.tconv(volume)
        output = torch.cat([output, cat], dim=1)
        output = self.activation(output)
        return output
