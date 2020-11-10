import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from Data import VideoFrameDataset
from Model import FrameClassifier


def test(ckpt_path, data_root, data_list, batch_size, num_workers, device, display_interval):
    ckpt = torch.load(ckpt_path)
    net_configs = ckpt['net_configs']
    print(f'Load ckpt from {ckpt_path}')

    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = VideoFrameDataset(
        root=data_root,
        video_list_path=data_list,
        transform=val_transform
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = FrameClassifier(
        cnn_type=net_configs['cnn_type'],
        fc_sizes=net_configs['hidden_sizes'],
        batchnorms=net_configs['batchnorms'],
        dropouts=net_configs['dropouts']
    )
    model.to(device)
    model.load_state_dict(ckpt['state_dict'])

    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        predicted_scores = []
        target_scores = []
        for step, (images, labels) in enumerate(tqdm(val_loader)):
            images = images.to(dtype=torch.float32, device=device)                  # (N, C, H, W)
            labels = labels.to(dtype=torch.float32, device=device)                  # (N,)

            # forward
            prob = model(images)                            # (N, 1)    after sigmoid

            n_correct += ((prob.detach().squeeze(1) >= 0.5) == labels).sum().item()
            n_samples += labels.shape[0]

            predicted_scores.append(prob.squeeze().detach().cpu().numpy())
            target_scores.append(labels.cpu().to(dtype=torch.int).numpy())

            # display
            if (step + 1) % display_interval == 0 and step < len(val_loader) - 1:
                tqdm.write('-' * 40)
                tqdm.write(f'[info] Probs: {prob.squeeze().detach().cpu().numpy()[:10]}')
                tqdm.write(f'[info] Label: {labels.cpu().numpy()[:10]}')

    print('=' * 80)
    print(f'[info] Test Acc = {n_correct / n_samples * 100:.4f}%')

    predicted_scores = np.concatenate(predicted_scores, axis=0)
    target_scores = np.concatenate(target_scores, axis=0)

    auc = roc_auc_score(target_scores, predicted_scores)
    print(f'[info] Test AUC = {auc:.5f}')


if __name__ == '__main__':
    test(
        ckpt_path='exp/20201110-10:29:10/m_epoch10.pt',
        data_root='/home/ubuntu/data',
        data_list='test.txt',
        batch_size=1024,
        num_workers=4,
        device='cuda',
        display_interval=1
    )
