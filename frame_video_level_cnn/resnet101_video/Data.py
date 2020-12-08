import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class VideoFrameDataset(Dataset):
    def __init__(self, root, video_list_path, transform):
        self.transform = transform

        with open(video_list_path, 'r') as fp:
            valid_videos = [line.rstrip().split()[0].split('/')[-1] for line in fp.readlines()]

        all_jpgs = sorted(glob.glob(root.rstrip('/') + '/*/*.jpg'))
        valid_jpgs = []
        for name in all_jpgs:
            if name.split('/')[-2] in valid_videos:
                valid_jpgs.append(name)

        self.images = valid_jpgs
        self.labels = [int(n.split('/')[-2].split('-')[-1] == '1') for n in valid_jpgs]

        tqdm.write(f'There are {len(self.images)} images, {sum(self.labels)} are anomaly.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.images[index]))
        return img, self.labels[index]



def loadImages(root, folder, batch, transform):
    valid_jpgs = sorted(glob.glob(root.rstrip('/') + '/' + str(folder) + '/*.jpg'))
    labels = [int(n.split('/')[-2].split('-')[-1] == '1') for n in valid_jpgs]

    valid_jpgs = [transform(Image.open(item)) for item in valid_jpgs]

    if len(valid_jpgs) > batch:
        valid_jpgs = valid_jpgs[:batch]
    elif len(valid_jpgs) < batch:
        for i in range(batch - len(valid_jpgs)):
            valid_jpgs.append(valid_jpgs[-1])

    return torch.stack(valid_jpgs, dim=0), torch.tensor(labels)
