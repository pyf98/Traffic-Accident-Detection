import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset


class RGBFrameDataset(Dataset):
    """Dataset class for 3-channel RGB images.
    """
    def __init__(self, root, video_list_path, n_frames, transform, is_train):
        super(RGBFrameDataset, self).__init__()
        self.root = root
        self.n_frames = n_frames
        self.transform = transform
        self.is_train = is_train

        with open(video_list_path, 'r') as fp:
            self.lines = [line.rstrip() for line in fp.readlines()]
        tqdm.write(f'[info] There are {len(self.lines)} videos in {video_list_path}')

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        """
        :param index: int
        :return result: (n_frames, C=3, H, W), label: 1 for anomaly, 0 for normal
        """
        line = self.lines[index]
        label = line.split()[1] == 'anomaly'        # anomaly: 1, normal: 0
        folder = os.path.join(self.root, line.split()[0])
        jpg_list = os.listdir(folder)
        jpg_list.sort()         # must sort to retain the order

        if len(jpg_list) > self.n_frames:      # there are enough frames
            if self.is_train:
                start = np.random.randint(0, len(jpg_list) - self.n_frames)
            else:
                start = 0
            jpg_list = jpg_list[start:start+self.n_frames]
        elif len(jpg_list) < self.n_frames:    # frames are not enough
            jpg_list += [jpg_list[-1]] * (self.n_frames - len(jpg_list))        # repeat the last frame

        assert len(jpg_list) == self.n_frames

        frames = []
        for jpg in jpg_list:
            image = Image.open(os.path.join(folder, jpg))
            image = self.transform(image)           # torch.Tensor, (C=3, H, W), range [0., 1.]
            frames.append(image)
        frames = torch.stack(frames, dim=0)         # (n_frames, C=3, H, W)

        return frames, label

    def collate_fn(self, batch):
        videos = torch.stack([b[0] for b in batch], dim=0)          # (batch_size, n_frames, C=3, H, W)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)      # (batch_size,)

        return videos, labels


class OpticalFlowDataset(Dataset):
    """Dataset class for stacked optical flow. Each optical flow image has two components, i.e. x and y.
    """
    def __init__(self, root, video_list_path, n_frames, transform, is_train):
        super(OpticalFlowDataset, self).__init__()

        self.root = root
        self.n_frames = n_frames
        self.transform = transform
        self.is_train = is_train

        with open(video_list_path, 'r') as fp:
            self.lines = [line.rstrip() for line in fp.readlines()]
        tqdm.write(f'[info] There are {len(self.lines)} videos in {video_list_path}')

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        """
        :param index: int
        :return frames: torch.FloatTensor, (n_frames * 2, H, W); label: 1 for anomaly, 0 for normal
        """
        line = self.lines[index]
        label = line.split()[1] == 'anomaly'        # anomaly: 1, normal: 0
        folder = os.path.join(self.root, line.split()[0])
        jpg_list = os.listdir(folder)
        jpg_list.sort()         # must sort to retain the order

        if len(jpg_list) > self.n_frames:      # there are enough frames
            if self.is_train:
                start = np.random.randint(0, len(jpg_list) - self.n_frames)
            else:
                start = 0
            jpg_list = jpg_list[start:start+self.n_frames]
        elif len(jpg_list) < self.n_frames:    # frames are not enough
            jpg_list += [jpg_list[-1]] * (self.n_frames - len(jpg_list))        # repeat the last frame

        assert len(jpg_list) == self.n_frames

        frames = []
        for jpg in jpg_list:
            image = Image.open(os.path.join(folder, jpg))   # (H, W, 3), channel 0: horizontal, channel 1: vertical
            image = self.transform(image)           # torch.FloatTensor, (3, H, W), range [0., 1.]
            image = image[:-1, :, :]                # (2, H, W)
            frames.append(image)
        frames = torch.cat(frames, dim=0)         # (n_frames * 2, H, W)
        assert frames.shape[0] == 2 * self.n_frames

        return frames, label

    def collate_fn(self, batch):
        videos = torch.stack([b[0] for b in batch], dim=0)          # (batch_size, n_frames * 2, H, W)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)      # (batch_size,)

        return videos, labels
