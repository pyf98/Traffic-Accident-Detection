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


def test(ckpt_path, data_root, data_list, batch_size, num_workers, device, display_interval):
    '''
    Video-level prediction
    '''
    ckpt = torch.load(ckpt_path)
    net_configs = ckpt['net_configs']
    print(f'Load ckpt from {ckpt_path}')

    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_videos = []
    with open(data_list, 'r') as fp:
        valid_videos = [line.rstrip().split()[0].split('/')[-1] for line in fp.readlines()]

    model = FrameClassifier(
        #cnn_type=net_configs['cnn_type'],
        fc_sizes=net_configs['hidden_sizes'],
        batchnorms=net_configs['batchnorms'],
        dropouts=net_configs['dropouts']
    )
    model.to(device)
    model.load_state_dict(ckpt['state_dict'])

    model.eval()
    with torch.no_grad():
        predicted_scores = []
        target_scores = []
        for i in range(len(valid_videos)):
            images, labels = loadImages(data_root, valid_videos[i], batch_size, val_transform)
            images = images.to(dtype=torch.float32, device=device)
            labels = labels.to(dtype=torch.float32, device=device)

            prob = model(images)

            np_prob = prob.squeeze().detach().cpu().numpy()
            np_target= labels.cpu().to(dtype=torch.int).numpy()

            predicted_avg_score = np.average(np_prob)
            target_avg_score = np.average(np_target)

            if (predicted_avg_score >= 0.5 and target_avg_score == 1.0) or (predicted_avg_score <= 0.5 and target_avg_score == 0.0):
                n_correct += 1

            predicted_scores.append(predicted_avg_score)
            target_scores.append(target_avg_score)

    print('=' * 80)
    print(f'[info] Test Acc = {n_correct / len(valid_videos) * 100:.4f}%')

    auc = roc_auc_score(target_scores, predicted_scores)
    print(f'[info] Test AUC = {auc:.5f}')


if __name__ == '__main__':
    test(
        ckpt_path='exp/20201201-09:14:02/m_epoch07.pt',
        data_root='/home/ubuntu/project/data',
        data_list='test.txt',
        batch_size=30,
        num_workers=4,
        device='cuda',
        display_interval=1
    )
