import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy, Precision, Recall
from tristar.models.action import ActionClassifier

class MultiLabelActionClassifier(ActionClassifier):
    def __init__(self, learning_rate=0.0001, rgb=True, depth=True, thermal=True):
        super(MultiLabelActionClassifier, self).__init__()

        self.learning_rate = learning_rate

        n_channels = 0

        if rgb:
            n_channels += 3
        if depth:
            n_channels += 1
        if thermal:
            n_channels += 1

        # Define layers
        self.features = nn.Sequential(
            # Block 1
            self.Conv3dBlock(n_channels, 64),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 2
            self.Conv3dBlock(64, 128),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 3
            self.Conv3dBlock(128, 256),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 4
            self.Conv3dBlock(256, 512),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # Change the output size to match the number of classes
            nn.Linear(512, 14),
            nn.Sigmoid()  # use sigmoid for multilabel classification
        )

    def Conv3dBlock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.binary_cross_entropy(outputs, targets)

        # Update metrics
        self.train_acc(outputs, targets)
        self.train_prec(outputs, targets)
        self.train_rec(outputs, targets)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.binary_cross_entropy(outputs, targets)

        # Update metrics
        self.val_acc(outputs, targets)
        self.val_prec(outputs, targets)
        self.val_rec(outputs, targets)

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.binary_cross_entropy(outputs, targets)

        # Update metrics
        self.test_acc(outputs, targets)
        self.test_prec(outputs, targets)
        self.test_rec(outputs, targets)

        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.test_acc.compute(), logger=True)
        self.log('test_prec_epoch', self.test_prec.compute(), logger=True)
        self.log('test_rec_epoch', self.test_rec.compute(), logger=True)

    def on_training_epoch_end(self):
        self.log('train_acc_epoch', self.train_acc.compute(), logger=True)
        self.log('train_prec_epoch', self.train_prec.compute(), logger=True)
        self.log('train_rec_epoch', self.train_rec.compute(), logger=True)

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.val_acc.compute(), logger=True)
        self.log('val_prec_epoch', self.val_prec.compute(), logger=True)
        self.log('val_rec_epoch', self.val_rec.compute(), logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
