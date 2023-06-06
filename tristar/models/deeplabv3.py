import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from typing import Any
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights, deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchmetrics import JaccardIndex

from .segmentation import HumanSegmentation


class HumanSegmentationDeepLabV3(HumanSegmentation):

    def __init__(self,pretrained=False, backbone = 'resnet50', learning_rate: float = 1e-4, rgb=True, depth=True, thermal=True):
        super(HumanSegmentationDeepLabV3, self).__init__()

        self.save_hyperparameters()

        n_channels = 0

        if rgb:
            n_channels += 3
        if depth:
            n_channels += 1
        if thermal:
            n_channels += 1

        self.learning_rate = learning_rate

        # Load the pre-trained model
        
        if backbone == 'resnet50':
            base_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, progress=True)
        elif backbone == 'resnet101':
            base_model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT, progress=True)

        # replace input layers
        old_weights = base_model.backbone.conv1.weight.data
        # Define new input layer with 5 channels
        new_conv = nn.Conv2d(n_channels, old_weights.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
        # Set weights of first 3 channels to pre-trained weights
        if rgb:
            new_conv.weight.data[:, :3, :, :] = old_weights.clone()

        # Copy weights from first two channels to last two new channels
        # new_conv.weight.data[:, 3:, :, :] = old_weights[:, :2, :, :].clone()
        # Replace input layer in model
        base_model.backbone.conv1 = new_conv

        # Similar to previous examples, copy weights from 'human' class to new output layer
        human_class_index = 0
        human_class_weights = base_model.classifier[4].weight.data[human_class_index].clone()
        base_model.classifier[4] = nn.Conv2d(
            base_model.classifier[4].in_channels, 1, kernel_size=1)
        base_model.classifier[4].weight.data[0] = human_class_weights

        # Freeze all layers
        # for param in base_model.parameters():
        #     param.requires_grad = False
        # # Unfreeze the first layer
        # for param in base_model.backbone.conv1.parameters():
        #     param.requires_grad = True

        self.model = base_model

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        return self.model(x)['out']

    # def on_epoch_start(self):
    #     if self.current_epoch > 0:
    #         print(f'updating lr: {self.learning_rate/5}')
    #         for param in self.model.parameters():
    #             param.requires_grad = True
    #         for g in self.trainer.optimizers[0].param_groups:
    #             g['lr'] = self.learning_rate / 5
