import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
from tristar.models.action_pytorch import ActionClassifier


class DependentActionClassifier(ActionClassifier):
    def __init__(self, learning_rate=0.0001, rgb=True, depth=True, thermal=True):
        super().__init__(learning_rate=learning_rate)

        n_channels = 0
        if rgb:
            n_channels += 3
        if depth:
            n_channels += 1
        if thermal:
            n_channels += 1

        # Define features layers
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

        # Classifier layers for dependent labels
        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Linear(512, 5),  # ['put_down', 'pick_up', 'drink', 'type', 'wave']
            nn.Sigmoid()
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Linear(512, 6),  # ['get_down', 'get_up', 'sit', 'walk', 'stand', 'lay']
            nn.Softmax(dim=1)
        )

        self.classifier3 = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Linear(512, 3),  # ['out_of_view', 'out_of_room', 'in_room']
            nn.Softmax(dim=1)
        )

        # Configure optimizer after model is built
        self.configure_optimizers()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x3 = self.classifier3(x)
        return x1, x2, x3

    def training_step(self, batch, batch_idx):
        # Move batch to device
        inputs, targets = batch
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)

        outputs1, outputs2, outputs3 = self(inputs)

        # Split targets into 3 groups
        targets1, targets2, targets3 = targets[:, :5], targets[:, 5:11], targets[:, 11:]

        # Calculate loss for each group
        loss1 = F.binary_cross_entropy(outputs1, targets1)
        loss2 = F.cross_entropy(outputs2, torch.max(targets2, 1)[1])
        loss3 = F.cross_entropy(outputs3, torch.max(targets3, 1)[1])
        loss = loss1 + loss2 + loss3

        # Update metrics
        outputs_all = torch.cat((outputs1, outputs2, outputs3), 1)
        targets_all = torch.cat((targets1, targets2, targets3), 1)

        self.train_acc(outputs_all, targets_all)
        self.train_prec(outputs_all, targets_all)
        self.train_rec(outputs_all, targets_all)
        self.train_f1(outputs_all, targets_all)

        return loss, {}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)

        outputs1, outputs2, outputs3 = self(inputs)

        targets1, targets2, targets3 = targets[:, :5], targets[:, 5:11], targets[:, 11:]

        loss1 = F.binary_cross_entropy(outputs1, targets1)
        loss2 = F.cross_entropy(outputs2, torch.max(targets2, 1)[1])
        loss3 = F.cross_entropy(outputs3, torch.max(targets3, 1)[1])
        loss = loss1 + loss2 + loss3

        outputs_all = torch.cat((outputs1, outputs2, outputs3), 1)
        targets_all = torch.cat((targets1, targets2, targets3), 1)

        self.val_acc(outputs_all, targets_all)
        self.val_prec(outputs_all, targets_all)
        self.val_rec(outputs_all, targets_all)
        self.val_f1(outputs_all, targets_all)

        return loss, {}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)

        outputs1, outputs2, outputs3 = self(inputs)

        targets1, targets2, targets3 = targets[:, :5], targets[:, 5:11], targets[:, 11:]

        loss1 = F.binary_cross_entropy(outputs1, targets1)
        loss2 = F.cross_entropy(outputs2, torch.max(targets2, 1)[1])
        loss3 = F.cross_entropy(outputs3, torch.max(targets3, 1)[1])
        loss = loss1 + loss2 + loss3

        outputs_all = torch.cat((outputs1, outputs2, outputs3), 1)
        targets_all = torch.cat((targets1, targets2, targets3), 1)

        self.test_acc(outputs_all, targets_all)
        self.test_prec(outputs_all, targets_all)
        self.test_rec(outputs_all, targets_all)
        self.test_f1(outputs_all, targets_all)

        return loss, {}
