from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from ..data.datasets import MultiModalDataset
from ..data.transforms import ActionListTransform, NormalizeListTransform, NormalizeTransform, MaskTransform


class ActionClassificationDataModule:
    """
    Pure PyTorch Data Module for Action Classification
    Replaces Lightning LightningDataModule
    """

    def __init__(self, data_dir: str = 'data', batch_size: int = 8, rgb=True, depth=True, thermal=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = 0  # Set to 0 to avoid multiprocessing issues on Windows
        self.rgb = rgb
        self.depth = depth
        self.thermal = thermal

        # Initialize datasets
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.setup()

    def setup(self):
        """Prepare datasets"""
        transform = transforms.Compose([
            NormalizeListTransform(rgb=self.rgb, depth=self.depth, thermal=self.thermal),
            transforms.Resize((320, 240), antialias=None)
        ])

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

        self.test_data = MultiModalDataset(
            self.data_dir, split='test', transform=transform,
            targets=['actions'],
            target_transform=ActionListTransform(),
            window_size=8,
            rgb=self.rgb, depth=self.depth, thermal=self.thermal
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class HumanSegmentationDataModule:
    """
    Pure PyTorch Data Module for Human Segmentation
    Replaces Lightning LightningDataModule
    """

    def __init__(self, data_dir: str = 'data', batch_size: int = 8, rgb=True, depth=True, thermal=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = 0
        self.rgb = rgb
        self.depth = depth
        self.thermal = thermal

        # Initialize datasets
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.setup()

    def setup(self):
        """Prepare datasets"""
        transform = transforms.Compose([
            NormalizeTransform(rgb=self.rgb, depth=self.depth, thermal=self.thermal),
            transforms.Resize((240, 320), antialias=None),
        ])
        mask_transform = transforms.Compose([
            MaskTransform(),
        ])

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

        self.test_data = MultiModalDataset(
            self.data_dir, split='test', transform=transform,
            targets=['mask'],
            target_transform=mask_transform,
            rgb=self.rgb, depth=self.depth, thermal=self.thermal
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
