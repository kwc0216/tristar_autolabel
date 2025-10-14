import click
import torch
import lightning as L
import os
import numpy as np
import random
import logging
from datetime import datetime

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from tristar.models.deeplabv3 import HumanSegmentationDeepLabV3
from tristar.models.multi_label_action import MultiLabelActionClassifier
from tristar.models.dependent_action import DependentActionClassifier
from tristar.models.unet import HumanSegmentationUnet
from tristar.data.dataloaders import ActionClassificationRGBDepthThermal, HumanSegmentationRGBDepthThermal
from tristar.callbacks import DetailedMetricsLogger


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
@click.option('--log_dir', default='logs', help='directory for training logs')
def train(task, architecture, rgb, depth, thermal, batch_size, learning_rate, epochs, checkpoint_out, log_dir):
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    # here to remove weird warning
    torch.set_float32_matmul_precision('medium')

    # Setup logging
    name = f"{task}-{architecture}-{rgb}-{depth}-{thermal}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_folder = os.path.join(log_dir, f"{name}_{timestamp}")
    os.makedirs(log_folder, exist_ok=True)

    # Configure file logging
    log_file = os.path.join(log_folder, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Log hyperparameters
    logger.info("="*50)
    logger.info(f"Training Configuration: {name}")
    logger.info("="*50)
    logger.info(f"Task: {task}")
    logger.info(f"Architecture: {architecture}")
    logger.info(f"RGB: {rgb}")
    logger.info(f"Depth: {depth}")
    logger.info(f"Thermal: {thermal}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Log Directory: {log_folder}")
    logger.info("="*50)

    # Initialize CSV logger for metrics
    csv_logger = CSVLogger(save_dir=log_folder, name='metrics')

    # Initialize our model

    if task == 'segmentation':
        data_module = HumanSegmentationRGBDepthThermal(
            data_dir='data/tristar',
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
            data_dir='data/tristar',
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

    # Define metrics logger callback
    metrics_logger = DetailedMetricsLogger(log_dir=log_folder)

    trainer = Trainer(
        max_epochs=epochs,
        devices=1,
        accelerator='gpu',
        logger=csv_logger,
        callbacks=[checkpoint_callback, metrics_logger],
        log_every_n_steps=1,
        enable_progress_bar=True
    )  # uses one GPU

    logger.info("Starting training...")
    trainer.fit(model, data_module)
    logger.info("Training completed!")
    logger.info(f"Checkpoints saved to: {os.path.join(checkpoint_out, name)}")
    logger.info(f"Logs saved to: {log_folder}")


if __name__ == '__main__':
    train()
