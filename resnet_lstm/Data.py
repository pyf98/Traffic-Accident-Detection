import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_sequence


class VideoDataset(Dataset):
    def __init__(self, root, video_list_path, max_frames=None, transform=None, is_train=True):
        self.root = root
        self.max_frames = max_frames
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
        :return result: (T, C=3, H, W), label: 1 for anomaly, 0 for normal
        """
        line = self.lines[index]
        label = line.split()[1] == 'anomaly'        # anomaly: 1, normal: 0
        folder = os.path.join(self.root, line.split()[0])
        jpg_list = os.listdir(folder)
        jpg_list.sort()         # must sort to retain the order

        if self.max_frames is not None and len(jpg_list) > self.max_frames:
            if self.is_train:
                start = np.random.randint(0, len(jpg_list) - self.max_frames)
            else:
                start = 0
            jpg_list = jpg_list[start:start+self.max_frames]

        result = []
        for jpg in jpg_list:
            image = Image.open(os.path.join(folder, jpg))

            if self.transform is not None:
                image = self.transform(image)       # torch.Tensor, (C=3, H, W)
            result.append(image)
        result = torch.stack(result, dim=0)         # (T, C=3, H, W)

        return result, label


def video_collate(batch):
    videos = [b[0] for b in batch]
    video_lens = torch.tensor([v.shape[0] for v in videos], dtype=torch.long)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)

    _, C, H, W = videos[0].shape
    max_len = video_lens.max()
    padded_videos = torch.zeros(size=[len(videos), max_len, C, H, W], dtype=torch.float32)
    for i, v in enumerate(videos):
        padded_videos[i, :v.shape[0], :, :, :] = v

    return padded_videos, video_lens, labels
