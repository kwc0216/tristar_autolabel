from typing import List
import torch
import numpy as np

# TODO: ask for jit, https://pytorch.org/vision/stable/transforms.html

class Threshold:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, x):
        return (x >= self.threshold).float()

class NormalizeTransform:
    def __init__(self, rgb=True, depth=True, thermal=True):
        self.mean_rgb = np.array([126.39476776123047,128.59066772460938,134.02708435058594])
        self.mean_depth = 2959.2861328125
        self.mean_thermal = 29903.11328125 
        self.std_rgb = np.array([84.279945,83.32872,82.45626])
        self.std_depth = 1928.0287860921578
        self.std_thermal = 149.91512572744287
        self.rgb = rgb
        self.depth = depth
        self.thermal = thermal

    def __call__(self, frames):
        tensors = []
        i = 0
        if self.rgb:  # RGB Image
            frame = (frames[i] - self.mean_rgb) / self.std_rgb
            frame = torch.from_numpy(frame).float().permute(2, 0, 1)
            tensors.append(frame)
            i+=1
        if self.depth:  # Depth Image
            frame = (frames[i] - self.mean_depth) / self.std_depth
            frame = torch.from_numpy(frame).float().unsqueeze(0)
            tensors.append(frame)
            i+=1
        if self.thermal:  # Thermal Image
            frame = (frames[i] - self.mean_thermal) / self.std_thermal
            frame = torch.from_numpy(frame).float().unsqueeze(0)
            tensors.append(frame)
            i+=1
        result = torch.cat(tensors, dim=0)
        return result


class NormalizeListTransform:
    def __init__(self, rgb=True, depth=True, thermal=True):
        self.normalize = NormalizeTransform(rgb, depth, thermal)
        self.rgb = rgb
        self.depth = depth
        self.thermal = thermal

    def __call__(self, all_frames):
        all_tensors = []
        for frames in all_frames:
            tensors = self.normalize(frames)
            all_tensors.append(tensors)
        all_tensors = torch.stack(all_tensors, dim=0)
        return all_tensors


class MaskTransform:
    def __call__(self, y: List[np.ndarray]):
        mask = y[0]
        mask = np.expand_dims(mask, axis=0) # add the channel dimension
        return torch.from_numpy(mask.astype(np.float32))


class ActionTransform:

    def __init__(self):
        self.actions = [
            'put_down', 'pick_up', 'drink', 'type', 'wave',
            'get_down', 'get_up',
            'sit', 'walk', 'stand', 'lay',
            'out_of_view', 'out_of_room', 'in_room'
        ]

    def __call__(self, y: List[np.ndarray]):
        labels = np.array([label in y[0] for label in self.actions])
        return torch.from_numpy(labels.astype(np.float32))


class MaskListTransform:
    def __call__(self, y: List[np.ndarray]):
        mask = y[0]
        return torch.from_numpy(mask.astype(np.float32))


class ActionListTransform:

    def __init__(self):
        self.actions = [
            'put_down', 'pick_up', 'drink', 'type', 'wave',
            'get_down', 'get_up',
            'sit', 'walk', 'stand', 'lay',
            'out_of_view', 'out_of_room', 'in_room'
        ]
        self.transform = ActionTransform()

    def __call__(self, y: List[List[np.ndarray]]):
        all_labels = y[1]
        all_labels = np.array([
            np.array([
                label in labels
                for label in self.actions
            ]) for labels in all_labels]
        )
        all_labels = np.any(all_labels, axis=0)
        return torch.from_numpy(all_labels.astype(np.float32))
