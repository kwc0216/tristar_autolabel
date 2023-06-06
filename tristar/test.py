import click
import torch
import lightning as L
import os

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from multimodal.models.deeplabv3 import HumanSegmentationDeepLabV3
from multimodal.models.multi_label_action import MultiLabelActionClassifier
from multimodal.models.movinet import ActionClassificationMoViNet
from multimodal.models.unet import HumanSegmentationUnet
from multimodal.data.dataloaders import ActionClassificationRGBDepthThermal, HumanSegmentationRGBDepthThermal

@click.command()
@click.option('--task', default='segmentation', help='Task to perform: segmentation or action_classification')
@click.option('--architecture', default='unet', help='Model architecture: unet, deeplabv3, vit for segmentation or r3d for action recognition')
@click.option('--rgb/--no-rgb', default=True, help='Use RGB input')
@click.option('--depth/--no-depth', default=True, help='Use depth input')
@click.option('--thermal/--no-thermal', default=True, help='Use thermal input')
@click.option('--checkpoint_dir', default='data/checkpoints', help='Directory of the checkpoint')
@click.option('--checkpoint_file', default='best_model.ckpt', help='File name of the checkpoint')
@click.option('--batch_size', default=16, help='Batch size for testing')
def test(task, architecture, rgb, depth, thermal, checkpoint_dir, checkpoint_file, batch_size):
    # here to remove weird warning
    torch.set_float32_matmul_precision('medium')

    # Initialize wandb logger
    name = f"{task}-{architecture}-{rgb}-{depth}-{thermal}"
    wandb_logger = WandbLogger(
        name=f"{task}-{architecture}-{rgb}-{depth}-{thermal}", project="multimodal"
    )

    # pass the logger to the Trainer
    trainer = L.Trainer(logger=wandb_logger)

    # Initialize our model

    if task == 'segmentation':
        data_module = HumanSegmentationRGBDepthThermal(
            batch_size=batch_size,
            rgb=rgb, depth=depth, thermal=thermal
        )
        if architecture == 'unet':
            model = HumanSegmentationUnet.load_from_checkpoint(
                checkpoint_path=os.path.join(checkpoint_dir, checkpoint_file),
                rgb=rgb, depth=depth, thermal=thermal
            )
        elif architecture == 'deeplabv3':
            model = HumanSegmentationDeepLabV3.load_from_checkpoint(
                checkpoint_path=os.path.join(checkpoint_dir, checkpoint_file),
                rgb=rgb, depth=depth, thermal=thermal
            )
    elif task == 'action_classification':
        data_module = ActionClassificationRGBDepthThermal(
            rgb=rgb, depth=depth, thermal=thermal
        )
        if architecture == 'action3d':
            model = MultiLabelActionClassifier.load_from_checkpoint(
                checkpoint_path=os.path.join(checkpoint_dir, checkpoint_file),
                rgb=rgb, depth=depth, thermal=thermal
            )
        else:
            raise ValueError('invalid architecture')
    else:
        raise ValueError('invalid task')

    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    test()
