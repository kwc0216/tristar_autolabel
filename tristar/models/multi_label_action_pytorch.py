import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
from tristar.models.action_pytorch import ActionClassifier


class MultiLabelActionClassifier(ActionClassifier):
    def __init__(self, learning_rate=0.0001, rgb=True, depth=True, thermal=True):
        super().__init__(learning_rate=learning_rate)

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

        # Configure optimizer after model is built
        self.configure_optimizers()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        # Move batch to device
        inputs, targets = batch
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)

        outputs = self(inputs)
        loss = F.binary_cross_entropy(outputs, targets)

        # Update metrics
        self.train_acc(outputs, targets)
        self.train_prec(outputs, targets)
        self.train_rec(outputs, targets)
        self.train_f1(outputs, targets)

        return loss, {}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)

        outputs = self(inputs)
        loss = F.binary_cross_entropy(outputs, targets)

        # Update metrics
        self.val_acc(outputs, targets)
        self.val_prec(outputs, targets)
        self.val_rec(outputs, targets)
        self.val_f1(outputs, targets)

        return loss, {}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)

        outputs = self(inputs)
        loss = F.binary_cross_entropy(outputs, targets)

        # Update metrics
        self.test_acc(outputs, targets)
        self.test_prec(outputs, targets)
        self.test_rec(outputs, targets)
        self.test_f1(outputs, targets)

        return loss, {}
