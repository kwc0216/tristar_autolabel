from torch.utils.data import DataLoader

import lightning as L
import torchvision.transforms as transforms

from ..data.datasets import MultiModalDataset
from ..data.transforms import ActionListTransform, NormalizeListTransform, NormalizeTransform, MaskTransform, Threshold

class HumanSegmentationRGBDepthThermal(L.LightningDataModule):

    def __init__(self, data_dir: str = 'data', batch_size: int = 8, rgb=True, depth=True, thermal=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = 16
        self.rgb = rgb
        self.depth = depth
        self.thermal = thermal

    def setup(self, stage: str):
        transform = transforms.Compose([
            NormalizeTransform(rgb=self.rgb, depth=self.depth, thermal=self.thermal),
            transforms.Resize((240, 320), antialias=None),
        ])
        mask_transform = transforms.Compose([
            MaskTransform(),
        ])
        if stage == "fit":
            self.train_data = MultiModalDataset(
                self.data_dir, split='train', transform=transform,
                targets=['mask'],
                target_transform=mask_transform,
                rgb=self.rgb, depth=self.depth, thermal=self.thermal
            )
            self.val_data = MultiModalDataset(
                self.data_dir, split='val', transform=transform,
                targets=['mask'],
                target_transform=mask_transform,
                rgb=self.rgb, depth=self.depth, thermal=self.thermal
            )
        if stage == "test":
            self.test_data = MultiModalDataset(
                self.data_dir, split='test', transform=transform,
                targets=['mask'],
                target_transform=mask_transform,
                rgb=self.rgb, depth=self.depth, thermal=self.thermal
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
        )
        return test_loader

class ActionClassificationRGBDepthThermal(L.LightningDataModule):

    def __init__(self, data_dir: str = 'data', batch_size: int = 8, rgb=True, depth=True, thermal=True):
        super().__init__()
        self.rgb = rgb
        self.depth = depth
        self.thermal = thermal
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = 16

    def setup(self, stage: str):
        transform = transforms.Compose([
            NormalizeListTransform(rgb=self.rgb, depth=self.depth, thermal=self.thermal),
            transforms.Resize((320, 240), antialias=None)
        ])
        if stage == "fit":
            self.train_data = MultiModalDataset(
                self.data_dir, split='train', transform=transform,
                targets=['actions'],
                target_transform=ActionListTransform(),
                window_size=8,
                rgb=self.rgb, depth=self.depth, thermal=self.thermal
            )
            self.val_data = MultiModalDataset(
                self.data_dir, split='val', transform=transform,
                targets=['actions'],
                target_transform=ActionListTransform(),
                window_size=8,
                rgb=self.rgb, depth=self.depth, thermal=self.thermal
            )
        if stage == "test":
            self.test_data = MultiModalDataset(
                self.data_dir, split='test', transform=transform,
                targets=['actions'],
                target_transform=ActionListTransform(),
                window_size=8,
                rgb=self.rgb, depth=self.depth, thermal=self.thermal
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
        )
        return test_loader

