import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights, deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchmetrics import JaccardIndex

from .segmentation_pytorch import HumanSegmentation


class HumanSegmentationDeepLabV3(HumanSegmentation):

    def __init__(self, pretrained=False, backbone='resnet50', learning_rate=1e-4, rgb=True, depth=True, thermal=True):
        super().__init__(learning_rate=learning_rate)

        n_channels = 0

        if rgb:
            n_channels += 3
        if depth:
            n_channels += 1
        if thermal:
            n_channels += 1

        # Load the pre-trained model
        if backbone == 'resnet50':
            base_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, progress=True)
        elif backbone == 'resnet101':
            base_model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT, progress=True)
        else:
            raise ValueError(f"Invalid backbone: {backbone}")

        # Replace input layers
        old_weights = base_model.backbone.conv1.weight.data
        # Define new input layer with n_channels
        new_conv = nn.Conv2d(n_channels, old_weights.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
        # Set weights of first 3 channels to pre-trained weights if using RGB
        if rgb:
            new_conv.weight.data[:, :3, :, :] = old_weights.clone()

        # Replace input layer in model
        base_model.backbone.conv1 = new_conv

        # Copy weights from 'human' class to new output layer
        human_class_index = 0
        human_class_weights = base_model.classifier[4].weight.data[human_class_index].clone()
        base_model.classifier[4] = nn.Conv2d(
            base_model.classifier[4].in_channels, 1, kernel_size=1)
        base_model.classifier[4].weight.data[0] = human_class_weights

        self.model = base_model

        # Configure optimizer after model is built
        self.configure_optimizers()

    def forward(self, x):
        return self.model(x)['out']
