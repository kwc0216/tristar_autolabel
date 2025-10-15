import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import JaccardIndex


class HumanSegmentation(nn.Module):
    """
    Base class for human segmentation models
    Pure PyTorch version (no Lightning dependency)
    """

    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self._device = 'cpu'

        # Initialize metrics
        self.train_iou = JaccardIndex(task='binary', num_classes=1)
        self.val_iou = JaccardIndex(task='binary', num_classes=1)
        self.test_iou = JaccardIndex(task='binary', num_classes=1)

        self.optimizer = None
        self.scheduler = None

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Optional: add scheduler
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.1, patience=5
        # )
        return self.optimizer

    def to(self, device):
        """Override to() to also move metrics to device"""
        super().to(device)
        self._device = device
        self.train_iou = self.train_iou.to(device)
        self.val_iou = self.val_iou.to(device)
        self.test_iou = self.test_iou.to(device)
        return self

    def forward(self, x):
        raise NotImplementedError

    def _calculate_loss(self, logits, y):
        """Calculate binary cross entropy loss"""
        y = F.interpolate(y, size=logits.shape[2:], mode='nearest')
        return F.binary_cross_entropy_with_logits(logits, y)

    def training_step(self, batch, batch_idx):
        """Training step - returns (loss, metrics_dict)"""
        x, y = batch
        x = x.to(self._device)
        y = y.to(self._device)

        logits = self(x)
        loss = self._calculate_loss(logits, y)

        # Calculate IoU
        if (logits < 0.5).all() and (y < 0.5).all():
            iou = torch.tensor(1.0, device=self._device)
        else:
            logits_upscaled = F.interpolate(
                logits, size=y.shape[2:], mode='bilinear', align_corners=False
            )
            iou = self.train_iou(logits_upscaled.sigmoid(), y)

        return loss, {'train_iou': iou}

    def validation_step(self, batch, batch_idx):
        """Validation step - returns (loss, metrics_dict)"""
        x, y = batch
        x = x.to(self._device)
        y = y.to(self._device)

        logits = self(x)
        loss = self._calculate_loss(logits, y)

        # Calculate IoU
        if (logits < 0.5).all() and (y < 0.5).all():
            iou = torch.tensor(1.0, device=self._device)
        else:
            logits_upscaled = F.interpolate(
                logits, size=y.shape[2:], mode='bilinear', align_corners=False
            )
            iou = self.val_iou(logits_upscaled.sigmoid(), y)

        if torch.isnan(iou).any().item():
            raise Exception('IoU cannot be NaN')

        return loss, {'val_iou': iou}

    def test_step(self, batch, batch_idx):
        """Test step - returns (loss, metrics_dict)"""
        x, y = batch
        x = x.to(self._device)
        y = y.to(self._device)

        logits = self(x)
        loss = self._calculate_loss(logits, y)

        # Calculate IoU
        if (logits < 0.5).all() and (y < 0.5).all():
            iou = torch.tensor(1.0, device=self._device)
        else:
            logits_upscaled = F.interpolate(
                logits, size=y.shape[2:], mode='bilinear', align_corners=False
            )
            iou = self.test_iou(logits_upscaled.sigmoid(), y)

        if torch.isnan(iou).any().item():
            raise Exception('IoU cannot be NaN')

        return loss, {'test_iou': iou}
