import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from Model import CRNNClassifier
from Data import VideoDataset, video_collate


def predict_video_anomaly_scores(frames_path, model, device, resize=224):
    """Predict an anomaly score for each frame in the input video.
    :return probs: numpy.ndarray of shape (T,), anomaly probs between 0. and 1.
    """

    transform = transforms.Compose([
        transforms.Resize([resize, resize]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load frames into a torch.Tensor of shape (N, T, C=3, H=224, W=224)
    img_list = [p for p in os.listdir(frames_path) if p.endswith('.jpg')]
    img_list.sort()     # must sort to retain the order

    images = []
    for name in img_list:
        image = Image.open(os.path.join(frames_path, name))
        image = transform(image)
        images.append(image)
    images = torch.stack(images, dim=0)                             # (T, C, H, W)
    images = images.unsqueeze(0).to(device=device)                  # (N=1, T, C=3, H=224, W=224)
    images_len = torch.tensor([images.shape[1]], dtype=torch.long, device=device)   # (N=1,)

    model.eval()
    with torch.no_grad():
        probs, probs_len = model.predict_probs(images, images_len)      # (N=1, T), (N,)
        probs = probs.squeeze(0).detach().cpu().numpy()                 # (T,)

    return probs
