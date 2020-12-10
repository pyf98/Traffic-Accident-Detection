import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from Data import VideoFrameDataset, loadImages
from Model import FrameClassifier
import matplotlib.pyplot as plt


def test(ckpt_path, data_root, data_list, batch_size, num_workers, device, display_interval):
    '''
    Genarates and saves the plot of 4 test videos
    '''
    ckpt = torch.load(ckpt_path)
    net_configs = ckpt['net_configs']
    print(f'Load ckpt from {ckpt_path}')

    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = FrameClassifier(
        #cnn_type=net_configs['cnn_type'],
        fc_sizes=net_configs['hidden_sizes'],
        batchnorms=net_configs['batchnorms'],
        dropouts=net_configs['dropouts']
    )
    model.to(device)
    model.load_state_dict(ckpt['state_dict'])

    valid_videos = [1, 2, 3, 4] # folder names

    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        predicted_scores = []
        target_scores = []
        for i in range(len(valid_videos)):
            images, labels = loadImages(data_root, valid_videos[i], batch_size, val_transform)
            images = images.to(dtype=torch.float32, device=device)
            labels = labels.to(dtype=torch.float32, device=device)

            prob = model(images)

            np_prob = prob.squeeze().detach().cpu().numpy()
            np_target= labels.cpu().to(dtype=torch.int).numpy()

            plt.plot(np.unique(np_prob))
            plt.ylabel('Anomaly Probability')
            plt.xlabel('Time')
            l = plt.axvline(x=int(len(np_prob)/2), linewidth=190, color='#FF5647', alpha=0.4)
            plt.grid(True)
            plt.savefig('plot' + str(i) + '.png')
            plt.clf()


if __name__ == '__main__':
    test(
        ckpt_path='exp/20201201-09:14:02/m_epoch07.pt',
        data_root='/home/ubuntu/project/data/plot',
        data_list='test.txt',
        batch_size=150,
        num_workers=4,
        device='cuda',
        display_interval=1
    )
