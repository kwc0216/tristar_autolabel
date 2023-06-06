import os
import glob
import cv2
import numpy as np
import pandas as pd
from typing import List
from torch.utils.data import Dataset


class MultiModalShotDataset(Dataset):

    def __init__(self,
                 root,
                 modalities: List[str] = ['rgb', 'depth', 'thermal'],
                 targets: List[str] = ['mask', 'actions'],
                 transform=None,
                 target_transform=None,
                 window_size=8) -> None:

        self.modalities = modalities
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.window_size = window_size

        files = {}
        last_modality = None

        for modality in modalities+['mask']:
            sample_path = os.path.join(root, modality)
            files[modality] = sorted(
                glob.glob(os.path.join(sample_path, '*')))
            if last_modality and files[modality] != files[last_modality]:
                raise Exception(f'ambitious data in {root}')

        self.files = files
        self.read_frame = {
            'rgb': lambda idx: cv2.imread(self.files['rgb'][idx], cv2.IMREAD_COLOR)[..., ::-1],
            'depth': lambda idx: cv2.imread(self.files['depth'][idx], cv2.IMREAD_ANYDEPTH),
            'thermal': lambda idx: cv2.imread(self.files['thermal'][idx], cv2.IMREAD_ANYDEPTH),
        }

        self.read_mask = lambda idx: np.load(self.files['mask'][idx])

        if 'actions' in self.targets:
            with open(os.path.join(root, 'actions.txt'), 'r') as f:
                self.actions = [line[:-1].split(' ')[1:]
                                for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.files[self.modalities[0]])

    def __getitem__(self, idx: int):
        if self.window_size == 1:
            frames = [
                self.read_frame[modality](idx)
                for modality in self.modalities
            ]
            labels = []
            if 'mask' in self.targets:
                labels.append(self.read_mask(idx))
            if 'actions' in self.targets:
                labels.append(self.actions[idx])
            if self.transform != None:
                frames = self.transform(frames)
            if self.target_transform != None:
                labels = self.target_transform(labels)
            return frames, labels
        else:
            all_frames = []
            masks = []
            actions = []
            for i in range(idx, idx+self.window_size):
                frames = [
                    self.read_frame[modality](i)
                    for modality in self.modalities
                ]
                all_frames.append(frames)
                if 'mask' in self.targets:
                    masks.append(self.read_mask(i))
                if 'actions' in self.targets:
                    actions.append(self.actions[i])
            labels = [masks, actions]
            if self.transform != None:
                all_frames = self.transform(all_frames)
            if self.target_transform != None:
                labels = self.target_transform(labels)
            return all_frames, labels


class MultiModalDataset(Dataset):

    def __init__(self, root='data/split',
                 split='train',
                 rgb=True, depth=True, thermal=True,
                 targets: List[str] = ['mask', 'actions'],
                 transform=None,
                 target_transform=None,
                 window_size=1):
        modalities: List[str] = []
        if rgb:
            modalities.append('rgb')
        if depth:
            modalities.append('depth')
        if thermal:
            modalities.append('thermal')
        assert all([modality in ['rgb', 'depth', 'thermal']
                   for modality in modalities])
        assert all([target in ['mask', 'actions'] for target in targets])
        assert split in ['train', 'validation', 'test', 'val']
        shots_dir = os.path.join(root, split)
        shot_dirs = [
            os.path.join(shots_dir, str(i))
            for i in range(len(os.listdir(shots_dir)))
        ]
        self.shots = [
            MultiModalShotDataset(
                shot_dir, modalities, targets, transform, target_transform, window_size
            )
            for shot_dir in shot_dirs
        ]
        self.window_size = window_size

    def __len__(self) -> int:
        return sum([len(shot)-self.window_size for shot in self.shots])

    def __getitem__(self, idx: int):
        cumulative_sum = 0
        for i, shot in enumerate(self.shots):
            shot_len = len(shot)-self.window_size+1
            if idx < cumulative_sum+shot_len:
                return self.shots[i][idx-cumulative_sum]
            cumulative_sum += shot_len+self.window_size-1
        raise Exception(f'{idx} out of bounds')
