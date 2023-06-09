import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall
import numpy as np
from typing import List
import os
from tristar.models.action import ActionClassifier


class DependentActionClassifier(ActionClassifier):
    def __init__(self, learning_rate=0.0001, rgb=True, depth=True, thermal=True):
        super().__init__()

        self.learning_rate = learning_rate

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

        # Overwrite classifier layer for dependent labels
        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),  # Adjust Dropout rate here
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),  # Adjust Dropout rate here
            nn.Linear(512, 5),  # ['put_down', 'pick_up', 'drink', 'type', 'wave']
            nn.Sigmoid()  # use sigmoid for multilabel classification
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),  # Adjust Dropout rate here
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),  # Adjust Dropout rate here
            nn.Linear(512, 6),  # ['get_down', 'get_up', 'sit', 'walk', 'stand', 'lay']
            nn.Softmax(dim=1)  # use softmax for mutually exclusive labels
        )
        
        self.classifier3 = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),  # Adjust Dropout rate here
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),  # Adjust Dropout rate here
            nn.Linear(512, 3),  # ['out_of_view', 'out_of_room', 'in_room']
            nn.Softmax(dim=1)  # use softmax for mutually exclusive labels
        )

        # self.classifier1 = nn.Sequential(
        #     nn.Linear(512 * 4 * 4 * 4, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(512, 5),  # ['put_down', 'pick_up', 'drink', 'type', 'wave']
        #     nn.Sigmoid()  # use sigmoid for multilabel classification
        # )
        # self.classifier2 = nn.Sequential(
        #     nn.Linear(512 * 4 * 4 * 4, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(512, 6),  # ['get_down', 'get_up', 'sit', 'walk', 'stand', 'lay']
        #     nn.Softmax(dim=1)  # use softmax for mutually exclusive labels
        # )
        # self.classifier3 = nn.Sequential(
        #     nn.Linear(512 * 4 * 4 * 4, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(512, 3),  # ['out_of_view', 'out_of_room', 'in_room']
        #     nn.Softmax(dim=1)  # use softmax for mutually exclusive labels
        # )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension

        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x3 = self.classifier3(x)
        return x1, x2, x3

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs1, outputs2, outputs3 = self(inputs)

        # Split targets into the 3 groups
        targets1, targets2, targets3 = targets[:, :5], targets[:, 5:11], targets[:, 11:]

        # Calculate loss for each group of outputs and sum them
        loss1 = F.binary_cross_entropy(outputs1, targets1)
        loss2 = F.cross_entropy(outputs2, torch.max(targets2, 1)[1])
        loss3 = F.cross_entropy(outputs3, torch.max(targets3, 1)[1])
        loss = loss1 + loss2 + loss3

        # Update metrics
        self.train_acc(torch.cat((outputs1, outputs2, outputs3), 1), torch.cat((targets1, targets2, targets3), 1))
        self.train_prec(torch.cat((outputs1, outputs2, outputs3), 1), torch.cat((targets1, targets2, targets3), 1))
        self.train_rec(torch.cat((outputs1, outputs2, outputs3), 1), torch.cat((targets1, targets2, targets3), 1))

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs1, outputs2, outputs3 = self(inputs)

        targets1, targets2, targets3 = targets[:, :5], targets[:, 5:11], targets[:, 11:]

        loss1 = F.binary_cross_entropy(outputs1, targets1)
        loss2 = F.cross_entropy(outputs2, torch.max(targets2, 1)[1])
        loss3 = F.cross_entropy(outputs3, torch.max(targets3, 1)[1])
        loss = loss1 + loss2 + loss3

        self.val_acc(torch.cat((outputs1, outputs2, outputs3), 1), torch.cat((targets1, targets2, targets3), 1))
        self.val_prec(torch.cat((outputs1, outputs2, outputs3), 1), torch.cat((targets1, targets2, targets3), 1))
        self.val_rec(torch.cat((outputs1, outputs2, outputs3), 1), torch.cat((targets1, targets2, targets3), 1))

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs1, outputs2, outputs3 = self(inputs)

        targets1, targets2, targets3 = targets[:, :5], targets[:, 5:11], targets[:, 11:]

        loss1 = F.binary_cross_entropy(outputs1, targets1)
        loss2 = F.cross_entropy(outputs2, torch.max(targets2, 1)[1])
        loss3 = F.cross_entropy(outputs3, torch.max(targets3, 1)[1])
        loss = loss1 + loss2 + loss3

        self.test_acc(torch.cat((outputs1, outputs2, outputs3), 1), torch.cat((targets1, targets2, targets3), 1))
        self.test_prec(torch.cat((outputs1, outputs2, outputs3), 1), torch.cat((targets1, targets2, targets3), 1))
        self.test_rec(torch.cat((outputs1, outputs2, outputs3), 1), torch.cat((targets1, targets2, targets3), 1))

        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01)  # weight_decay is added
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
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

        return {
           'optimizer': optimizer,
           'lr_scheduler': {
               'scheduler': scheduler,
               'monitor': 'val_loss',  # Assumes you have a validation loss logged
           }
        }


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
