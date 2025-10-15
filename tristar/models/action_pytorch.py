import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score


class ActionClassifier(nn.Module):
    """
    Pure PyTorch Action Classifier Base Class
    Replaces Lightning LightningModule
    """

    def __init__(self, num_labels=14, learning_rate=0.0001):
        super(ActionClassifier, self).__init__()
        self.num_labels = num_labels
        self.learning_rate = learning_rate

        # Define metrics
        self.train_acc = Accuracy(num_labels=num_labels, task='multilabel')
        self.train_prec = Precision(num_labels=num_labels, task='multilabel')
        self.train_rec = Recall(num_labels=num_labels, task='multilabel')
        self.train_f1 = F1Score(num_labels=num_labels, task='multilabel')

        self.val_acc = Accuracy(num_labels=num_labels, task='multilabel')
        self.val_prec = Precision(num_labels=num_labels, task='multilabel')
        self.val_rec = Recall(num_labels=num_labels, task='multilabel')
        self.val_f1 = F1Score(num_labels=num_labels, task='multilabel')

        self.test_acc = Accuracy(num_labels=num_labels, task='multilabel')
        self.test_prec = Precision(num_labels=num_labels, task='multilabel')
        self.test_rec = Recall(num_labels=num_labels, task='multilabel')
        self.test_f1 = F1Score(num_labels=num_labels, task='multilabel')

        # Optimizer will be set after model is fully initialized
        self.optimizer = None
        self.scheduler = None

        # Device tracking
        self._device = torch.device('cpu')

    def Conv3dBlock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def to(self, device):
        """Override to method to track device"""
        self._device = device
        # Move metrics to device
        self.train_acc = self.train_acc.to(device)
        self.train_prec = self.train_prec.to(device)
        self.train_rec = self.train_rec.to(device)
        self.train_f1 = self.train_f1.to(device)

        self.val_acc = self.val_acc.to(device)
        self.val_prec = self.val_prec.to(device)
        self.val_rec = self.val_rec.to(device)
        self.val_f1 = self.val_f1.to(device)

        self.test_acc = self.test_acc.to(device)
        self.test_prec = self.test_prec.to(device)
        self.test_rec = self.test_rec.to(device)
        self.test_f1 = self.test_f1.to(device)

        return super().to(device)

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

        return optimizer, scheduler

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """Training step - returns loss and metrics"""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """Validation step - returns loss and metrics"""
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        """Test step - returns loss and metrics"""
        raise NotImplementedError
