import click
import torch
import lightning as L
import os
import numpy as np
import random

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from tristar.models.deeplabv3 import HumanSegmentationDeepLabV3
from tristar.models.multi_label_action import MultiLabelActionClassifier
from tristar.models.dependent_action import DependentActionClassifier
from tristar.models.unet import HumanSegmentationUnet
from tristar.data.dataloaders import ActionClassificationRGBDepthThermal, HumanSegmentationRGBDepthThermal


@click.command()
@click.option('--task', default='segmentation', help='Task to perform: segmentation or action_recognition')
@click.option('--architecture', default='unet', help='Model architecture: unet, deeplabv3, vit for segmentation or r3d for action recognition')
@click.option('--rgb/--no-rgb', default=True, help='Use RGB input')
@click.option('--depth/--no-depth', default=True, help='Use depth input')
@click.option('--thermal/--no-thermal', default=True, help='Use thermal input')
@click.option('--batch_size', default=8, help='Batch size for training')
@click.option('--learning_rate', default=0.0001, help='Learning rate for the optimizer')
@click.option('--epochs', default=10, help='Number of epochs for training')
@click.option('--checkpoint_out', default='data/checkpoints', help='output path of the checkpoint')
def train(task, architecture, rgb, depth, thermal, batch_size, learning_rate, epochs, checkpoint_out):
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    
    # here to remove weird warning
    torch.set_float32_matmul_precision('medium')

    # initialize wandb logger
    name = f"{task}-{architecture}-{rgb}-{depth}-{thermal}"
    wandb_logger = WandbLogger(
        name=f"{task}-{architecture}-{rgb}-{depth}-{thermal}", project="multimodal"
    )
    wandb_logger.log_hyperparams({
        'task': task,
        'architecture': architecture,
        'rgb': rgb,
        'depth': depth,
        'thermal': thermal,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs
    })

    # pass the logger to the Trainer
    trainer = L.Trainer(logger=wandb_logger)

    # Initialize our model

    if task == 'segmentation':
        data_module = HumanSegmentationRGBDepthThermal(
            batch_size=batch_size,
            rgb=rgb, depth=depth, thermal=thermal
        )
        if architecture == 'unet':
            model = HumanSegmentationUnet(
                learning_rate=learning_rate,
                rgb=rgb, depth=depth, thermal=thermal
            )
        elif architecture == 'deeplabv3':
            model = HumanSegmentationDeepLabV3(
                learning_rate=learning_rate,
                rgb=rgb, depth=depth, thermal=thermal
            )
    elif task == 'action_classification':
        data_module = ActionClassificationRGBDepthThermal(
            batch_size=batch_size,
            rgb=rgb, depth=depth, thermal=thermal
        )
        if architecture == 'multilabel':
            model = MultiLabelActionClassifier(
                learning_rate=learning_rate,
                rgb=rgb, depth=depth, thermal=thermal
            )
        elif architecture == 'dependent':
            model = DependentActionClassifier(
                learning_rate=learning_rate,
                rgb=rgb, depth=depth, thermal=thermal
            )
        else:
            raise ValueError('invalid architecture')
    else:
        raise ValueError('invalid task')

    # define a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(checkpoint_out, name),
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = Trainer(
        max_epochs=epochs,
        devices=1,
        accelerator='gpu',
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )  # uses one GPU
    trainer.fit(model, data_module)


if __name__ == '__main__':
    train()
