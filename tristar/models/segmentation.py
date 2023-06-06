import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from typing import Any
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights, deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchmetrics import JaccardIndex


class HumanSegmentation(L.LightningModule):

    def __init__(self):
        super().__init__()

        self.train_iou = JaccardIndex('binary', num_classes=1)
        self.val_iou = JaccardIndex('binary', num_classes=1)
        self.test_iou = JaccardIndex('binary', num_classes=1)

    def on_epoch_start(self):
        # After the first epoch, unfreeze all layers and lower the learning rate
        if self.current_epoch > 0:
            for param in self.base_model.parameters():
                param.requires_grad = True
            self.learning_rate /= 10

    def forward(self, x):
        raise NotImplementedError

    def _calculate_loss(self, logits, y):
        y = F.interpolate(y, size=logits.shape[2:], mode='nearest')
        return F.binary_cross_entropy_with_logits(logits, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = y.unsqueeze(1)  # Add channel dimension to match output
        logits = self(x)
        loss = self._calculate_loss(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        if (logits<0.5).all() and (y<0.5).all():
            iou = torch.tensor(1)
        else:
            logits_upscaled = torch.nn.functional.interpolate(logits, size=y.shape[2:], mode='bilinear', align_corners=False)
            iou = self.train_iou(logits_upscaled.sigmoid(), y)
        self.log('train_iou', iou, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # y = y.unsqueeze(1)  # Add channel dimension to match output

        logits = self(x)
        loss = self._calculate_loss(logits, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        if (logits<0.5).all() and (y<0.5).all():
            iou = torch.tensor(1)
        else:
            logits_upscaled = torch.nn.functional.interpolate(logits, size=y.shape[2:], mode='bilinear', align_corners=False)
            iou = self.val_iou(logits_upscaled.sigmoid(), y)
        if torch.isnan(iou).any().item():
            raise Exception('iou cannot be nan')
        self.log('val_iou', iou, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True)

        if (logits<0.5).all() and (y<0.5).all():
            iou = torch.tensor(1)
        else:
            logits_upscaled = torch.nn.functional.interpolate(logits, size=y.shape[2:], mode='bilinear', align_corners=False)
            iou = self.test_iou(logits_upscaled.sigmoid(), y)
        if torch.isnan(iou).any().item():
            raise Exception('iou cannot be nan')
        self.log('test_iou', iou, on_step=True, on_epoch=True)
        return loss   

    def on_epoch_start(self):
        if self.current_epoch > 0:
            for param in self.model.parameters():
                param.requires_grad = True
            for g in self.trainer.optimizers[0].param_groups:
                g['lr'] = self.learning_rate / 5
