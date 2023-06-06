import click
import torch
import lightning as L
import os
import glob
import re
import json

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from multimodal.models.deeplabv3 import HumanSegmentationDeepLabV3
from multimodal.models.multi_label_action import MultiLabelActionClassifier
from multimodal.models.movinet import ActionClassificationMoViNet
from multimodal.models.unet import HumanSegmentationUnet
from multimodal.models.dependent_action import DependentActionClassifier
from multimodal.data.dataloaders import ActionClassificationRGBDepthThermal, HumanSegmentationRGBDepthThermal
import wandb


def test(task, architecture, rgb, depth, thermal, checkpoint_file, batch_size):
    # here to remove weird warning
    torch.set_float32_matmul_precision('medium')

    # Initialize wandb logger
    name = f"{task}-{architecture}-{rgb}-{depth}-{thermal}"
    wandb_run = wandb.init(project="multimodal", name=name)
    # wandb_logger = WandbLogger(
    #     name=f"test-{task}-{architecture}-{rgb}-{depth}-{thermal}", project="multimodal"
    # )

    # pass the logger to the Trainer
    trainer = L.Trainer(logger=WandbLogger())

    # Initialize our model

    if task == 'segmentation':
        data_module = HumanSegmentationRGBDepthThermal(
            batch_size=batch_size,
            rgb=rgb, depth=depth, thermal=thermal
        )
        if architecture == 'unet':
            model = HumanSegmentationUnet.load_from_checkpoint(
                checkpoint_path=checkpoint_file,
                rgb=rgb, depth=depth, thermal=thermal
            )
        elif architecture == 'deeplabv3':
            model = HumanSegmentationDeepLabV3.load_from_checkpoint(
                checkpoint_path=checkpoint_file,
                rgb=rgb, depth=depth, thermal=thermal
            )
    elif task == 'action_classification':
        data_module = ActionClassificationRGBDepthThermal(
            rgb=rgb, depth=depth, thermal=thermal
        )
        if architecture == 'dependent':
            model = DependentActionClassifier.load_from_checkpoint(
                checkpoint_path= checkpoint_file,
                rgb=rgb, depth=depth, thermal=thermal
            )
        elif architecture == 'action3d':
            model = MultiLabelActionClassifier.load_from_checkpoint(
                checkpoint_path= checkpoint_file,
                rgb=rgb, depth=depth, thermal=thermal
            )
        else:
            raise ValueError('invalid architecture')
    else:
        raise ValueError('invalid task')

    test_results = trainer.test(model, datamodule=data_module)
    
    # Write test results to a json file
    with open(f"{name}_test_results.json", 'w') as outfile:
        json.dump(test_results, outfile)
    # wandb_logger.finalize('success')

    wandb_run.finish()


@click.command()
@click.option('--checkpoint_dir', default='data/checkpoints', help='Directory of the checkpoint')
def find_and_test_best_models(checkpoint_dir):
    # Find all .ckpt files in the directory
    ckpt_files = glob.glob(f'{checkpoint_dir}/*/*.ckpt')

    # Keep track of the best loss and best model for each task-architecture-modality combination
    best_models = {}

    # Regex pattern to extract loss value from filename
    loss_pattern = re.compile(r"val_loss=([0-9.]+)\.ckpt")

    for file in ckpt_files:
        # Extract the loss from the filename using regex
        match = loss_pattern.search(file)
        if match is not None:
            loss = float(match.group(1))

            # Extract task-architecture-modality combination from the directory name
            combination = os.path.dirname(file)[len(checkpoint_dir)+1:]

            # Check if this combination is new or if this model has a lower loss than the current best
            if combination not in best_models or loss < best_models[combination][0]:
                best_models[combination] = (loss, file)

    if not best_models:
        print("No models found.")
    else:
        for combination, (loss, model) in best_models.items():
            print(f"Best model for {combination} is: {model} with a loss of {loss}")

            # Extract parameters from the folder name
            task, architecture, rgb, depth, thermal = combination.split('-')
            
            # Convert string Boolean to actual Boolean
            rgb = rgb == 'True'
            depth = depth == 'True'
            thermal = thermal == 'True'
            
            # Call test function
            #    task, architecture, rgb, depth, thermal, checkpoint_dir, checkpoint_file, batch_size
            test(task, architecture, rgb, depth, thermal, model, 8)


if __name__ == '__main__':
    find_and_test_best_models()
