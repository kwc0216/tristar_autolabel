import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy, Precision, Recall, F1Score



class ActionClassifier(L.LightningModule):
    def __init__(self, num_labels=14):
        super(ActionClassifier, self).__init__()
        self.num_labels=num_labels

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

    def Conv3dBlock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

