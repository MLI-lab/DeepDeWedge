# %%
import ast
import inspect
import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import typer
import torch
from typer_config import conf_callback_factory
from typing import Union, List
from typing_extensions import Annotated

from .utils.dataloader import MultiEpochsDataLoader as DataLoader
from .utils.load_function_args_from_yaml_config import \
    load_function_args_from_yaml_config
from .utils.subtomo_dataset import SubtomoDataset
from .utils.unet import LitUnet3D


loader = lambda yaml_config_file: load_function_args_from_yaml_config(
    function=fit_model, yaml_config_file=yaml_config_file
)
callback = conf_callback_factory(loader)


def fit_model(
    unet_params_dict: Annotated[
        str,
        typer.Option(
            callback=ast.literal_eval,
            help=f"Dictionary of parameters for the U-Net model. See {inspect.getfile(LitUnet3D)} for details.",
        ),
    ],
    adam_params_dict: Annotated[
        str,
        typer.Option(
            callback=ast.literal_eval,
            help="Dictionary of parameters for PyTroch's the Adam optimizer. See the tutorial notebook on model fitting or 'https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam' for details.",
        ),
    ],
    num_epochs: Annotated[int, typer.Option(help="Number of epochs to fit the model.")],
    batch_size: Annotated[int, typer.Option(help="Batch size for the optimizer.")],
    subtomo_size: Annotated[
        int, typer.Option(help="Size of the cubic subtomograms used for model fitting.")
    ],
    mw_angle: Annotated[
        float, typer.Option(help="Width of the missing wedge in degrees.")
    ],
    gpu: Annotated[List[int], typer.Option(help="Which GPU(s) to use for model fitting. Example: gpu=0 uses the first GPU, gpu=[0,1] uses the first two GPUs.")],
    num_workers: Annotated[
        int,
        typer.Option(
            help="Number of CPU workers to use for data loading. If fitting is slow, try increasing this number."
        ),
    ],
    subtomo_dir: Annotated[
        Optional[str],
        typer.Option(
            help="Path to the directory containing the subtomograms. If subtomo_dir is not provided, subtomo_dir is set to '{project_dir}/subtomos'."
        ),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option(
            help="If either subtomo_dir or logdir is not provided, project_dir must be provided, and the missing directory will be set to '{project_dir}/subtomos' or '{project_dir}/logs'."
        ),
    ] = None,
    logdir: Annotated[
        Optional[str],
        typer.Option(
            help="Path to the directory where the model checkpoints and logs will be saved. If logdir is not provided, logdir is set to '{project_dir}/logs'."
        ),
    ] = None,
    logger: Annotated[
        str,
        typer.Option(
            help="Which PyTorch Lightning logger to use. Choose from 'tensorboard' or 'csv'. See also 'https://lightning.ai/docs/pytorch/stable/extensions/logging.html'."
        ),
    ] = "tensorboard",
    check_val_every_n_epochs: Annotated[
        int, typer.Option(help="Check validation loss every n epochs.")
    ] = 10,
    update_subtomo_missing_wedges_every_n_epochs: Annotated[
        int,
        typer.Option(
            help="After how many epochs to update the missing wedge in the subtomograms."
        ),
    ] = 10,
    save_model_every_n_epochs: Annotated[
        int, typer.Option(help="Save a model checkpoint to logdir every n epochs.")
    ] = 10,
    save_n_models_with_lowest_fitting_loss: Annotated[
        int,
        typer.Option(help="Save the n models with the lowest fitting loss to logdir."),
    ] = 5,
    save_n_models_with_lowest_val_loss: Annotated[
        int,
        typer.Option(
            help="Save the n models with the lowest validation loss to logdir."
        ),
    ] = 5,
    resume_from_checkpoint: Annotated[
        Optional[str], typer.Option(help="Continue model fitting from a checkpoint.")
    ] = None,
    distributed_backend: Annotated[
        str, 
        typer.Option(help="Distributed backend to use when fitting on multiple GPUs, e.g, 'nccl' (default) or 'gloo'. Ignored if fitting on a single GPU.")
    ] = "nccl",
    seed: Annotated[
        Optional[int], typer.Option(help="Seed for reproducibility.")
    ] = None,
    config: Annotated[
        Optional[str],
        typer.Option(
            callback=callback,
            is_eager=True,
            help="Path to a yaml file containing the argumens for this function. Comand line arguments will overwrite the ones in the yaml file.",
        ),
    ] = None,
):
    """
    Fit a U-Net model for denoising and missing wedge reconstruction on sub-tomograms. Typically run after `prepare-data`.
    """
    pl.seed_everything(seed, workers=True)
    # setup subtomo_dir
    if subtomo_dir is None:
        if project_dir is not None:
            subtomo_dir = f"{project_dir}/subtomos"
        else:
            raise ValueError(
                "If project_dir is not provided, subtomo_dir must be provided."
            )
    # setup logdir
    if logdir is None:
        if project_dir is not None:
            logdir = f"{project_dir}/logs"
        else:
            raise ValueError("If project_dir is not provided, logdir must be provided.")
    logdir = Path(logdir)
    # setup logger
    if not os.path.exists(logdir.parent):
        os.makedirs(logdir.parent)
    if logger == "tensorboard":
        logger = pl.loggers.TensorBoardLogger(logdir.parent, name=logdir.name)
    elif logger == "csv":
        logger = pl.loggers.CSVLogger(logdir.parent, name=logdir.name)
    else:
        raise ValueError(
            f"Logger '{logger}' not recognized. Choose from 'tensorboard' or 'csv'."
        )
    logdir = f"{logger.save_dir}/{logger.name}/version_{logger.version}"
    print(f"Saving logs and model checkpoints to '{logdir}'")

    # check if there are subtomos for validation
    val_data_exists = (
        os.path.exists(f"{subtomo_dir}/val_subtomos")
        and len(os.listdir(f"{subtomo_dir}/val_subtomos/subtomo0")) > 0
    )
    if not val_data_exists:
        print(
            "Running model fitting without validation, as no validation data was found!"
        )

    if not subtomo_size % (2 ** unet_params_dict["num_downsample_layers"]) == 0:
        raise ValueError(
            f"subtomo_size must be divisible by 2^unet_params_dict['num_downsample_layers'] to ensure compatibility with the U-Net architecture. "
            f"Got subtomo_size={subtomo_size} and num_downsample_layers={unet_params_dict['num_downsample_layers']}."
        )

    # setup datasets
    fitting_dataset = SubtomoDataset(
        subtomo_dir=f"{subtomo_dir}/fitting_subtomos",
        crop_subtomos_to_size=subtomo_size,
        mw_angle=mw_angle,
        rotate_subtomos=True,
        deterministic_rotations=False,
    )
    if val_data_exists:
        val_dataset = SubtomoDataset(
            subtomo_dir=f"{subtomo_dir}/val_subtomos",
            crop_subtomos_to_size=subtomo_size,
            mw_angle=mw_angle,
            rotate_subtomos=True,
            deterministic_rotations=True,
        )
    # setup callbacks
    callbacks = []
    # lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    # callbacks.append(lr_callback)
    # this saves the model everey 50 epochs
    epoch_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{logdir}/checkpoints/epoch",
        filename="{epoch}",
        monitor="epoch",
        verbose=True,
        save_top_k=-1,
        every_n_epochs=save_model_every_n_epochs,
        save_on_train_epoch_end=True,
    )
    callbacks.append(epoch_callback)
    # this saves the best model based on the training loss
    if save_n_models_with_lowest_fitting_loss > 0:
        fitting_loss_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"{logdir}/checkpoints/fitting_loss",
            filename="{epoch}-{fitting_loss:.5f}",
            monitor="fitting_loss",
            verbose=True,
            save_top_k=save_n_models_with_lowest_fitting_loss,
            save_on_train_epoch_end=True,
        )
        callbacks.append(fitting_loss_callback)
    # this saves the top 3 models with the lowest validation loss
    if save_n_models_with_lowest_val_loss > 0 and val_data_exists:
        val_loss_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"{logdir}/checkpoints/val_loss",
            filename="{epoch}-{val_loss:.5f}",
            monitor="val_loss",
            verbose=True,
            save_top_k=save_n_models_with_lowest_val_loss,
        )
        callbacks.append(val_loss_callback)
    # initialize the model
    lit_unet = LitUnet3D(
        unet_params=unet_params_dict,
        adam_params=adam_params_dict,
        subtomo_dir=subtomo_dir,
        update_subtomo_missing_wedges_every_n_epochs=update_subtomo_missing_wedges_every_n_epochs,
    )
    # initialize the trainer
    devices = [gpu] if isinstance(gpu, int) else gpu
    strategy = pl.strategies.DDPStrategy(
        process_group_backend=distributed_backend, 
        find_unused_parameters=False,  # setting this to true gave a warning that it might slow things down
    ) if len(devices) > 1 else None
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        check_val_every_n_epoch=(
            check_val_every_n_epochs if val_data_exists else num_epochs
        ),
        deterministic=True,
        logger=logger,
        callbacks=callbacks,
        detect_anomaly=True,
        resume_from_checkpoint=resume_from_checkpoint,  # for pytorch-lightning < 2.0
    )

    # setup dataloaders
    fitting_dataloader = DataLoader(
        dataset=fitting_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    if val_data_exists:
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
    else:
        val_dataloader = None
    # fit the model
    if val_data_exists and resume_from_checkpoint is None:
        trainer.validate(lit_unet, val_dataloader)
    trainer.fit(
        #ckpt_path=resume_from_checkpoint,  # for pytorch-lightning >= 2.0
        model=lit_unet,
        train_dataloaders=fitting_dataloader,
        val_dataloaders=val_dataloader,
    )
