import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from typing import Any
from torchmetrics import JaccardIndex

from movinets import MoViNet
from movinets.config import _C

class ActionClassificationMoViNet(L.LightningModule):

    def __init__(self, learning_rate=0.0001):
        super().__init__()
        cfg = _C.MODEL.MoViNetA0
        cfg.conv1.input_channels = 5
        base_model = MoViNet(cfg, causal=False, pretrained=False)
        # 51
        base_model.classifier[3] = nn.Conv3d(2048, 14, (1,1,1))
        self.learning_rate = learning_rate
        self.model=base_model

    def forward(self, x):
        raise self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        #y = y.unsqueeze(1)  # Add channel dimension to match output
        logits = self.model(x.permute((0,2,1,3,4)))
        # y = y.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x.permute((0,2,1,3,4)))
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

