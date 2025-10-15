import click
import torch
import os
import numpy as np
import random
import logging
from datetime import datetime

from tristar.models.dependent_action_pytorch import DependentActionClassifier
from tristar.data.dataloaders_pytorch import ActionClassificationDataModule, HumanSegmentationDataModule
from tristar.callbacks.metrics_logger_pytorch import DetailedMetricsLogger, ModelCheckpoint
from tristar.trainer.pytorch_trainer import PyTorchTrainer


@click.command()
@click.option('--task', default='segmentation', help='Task to perform: segmentation or action_classification')
@click.option('--architecture', default='unet', help='Model architecture: unet, deeplabv3 for segmentation or multilabel, dependent for action classification')
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

    # Initialize our model and data module
    if task == 'segmentation':
        data_module = HumanSegmentationDataModule(
            data_dir='data/tristar',
            batch_size=batch_size,
            rgb=rgb, depth=depth, thermal=thermal
        )
        if architecture == 'unet':
            # Will be created in next step
            from tristar.models.unet_pytorch import HumanSegmentationUnet
            model = HumanSegmentationUnet(
                learning_rate=learning_rate,
                rgb=rgb, depth=depth, thermal=thermal
            )
        elif architecture == 'deeplabv3':
            # Will be created in next step
            from tristar.models.deeplabv3_pytorch import HumanSegmentationDeepLabV3
            model = HumanSegmentationDeepLabV3(
                learning_rate=learning_rate,
                rgb=rgb, depth=depth, thermal=thermal
            )
        else:
            raise ValueError('invalid architecture')
    elif task == 'action_classification':
        data_module = ActionClassificationDataModule(
            data_dir='data/tristar',
            batch_size=batch_size,
            rgb=rgb, depth=depth, thermal=thermal
        )
        if architecture == 'multilabel':
            # Will be created in next step
            from tristar.models.multi_label_action_pytorch import MultiLabelActionClassifier
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

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_out, name),
        monitor='val_loss',
        save_top_k=3,
        mode='min',
    )

    # Define metrics logger callback
    metrics_logger = DetailedMetricsLogger(log_dir=log_folder)

    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = PyTorchTrainer(
        model=model,
        max_epochs=epochs,
        device=device,
        callbacks=[checkpoint_callback, metrics_logger],
        checkpoint_dir=os.path.join(checkpoint_out, name)
    )

    logger.info("Starting training...")
    trainer.fit(data_module.train_dataloader(), data_module.val_dataloader())
    logger.info("Training completed!")
    logger.info(f"Checkpoints saved to: {os.path.join(checkpoint_out, name)}")
    logger.info(f"Logs saved to: {log_folder}")

    # Run test if available
    if data_module.test_data:
        logger.info("Starting testing...")
        trainer.test(data_module.test_dataloader())
        logger.info("Testing completed!")


if __name__ == '__main__':
    train()
