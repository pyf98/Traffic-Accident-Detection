import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from data import OpticalFlowDataset


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    """Test the CNN model on optical flow images and save results as a numpy file.
    """
    ckpt = torch.load(args.ckpt)

    transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
    ])
    dataset = OpticalFlowDataset(
        root=args.data_root,
        video_list_path=args.data_list,
        n_frames=ckpt['net_configs']['n_frames'],
        transform=transform,
        is_train=False
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )

    assert ckpt['net_configs']['cnn_type'] in ['resnet50', 'resnet101', 'resnet152']
    if ckpt['net_configs']['cnn_type'] == 'resnet50':
        from networks import resnet50 as ResNet
    elif ckpt['net_configs']['cnn_type'] == 'resnet101':
        from networks import resnet101 as ResNet
    elif ckpt['net_configs']['cnn_type'] == 'resnet152':
        from networks import resnet152 as ResNet

    model = ResNet(pretrained=True, channel=ckpt['net_configs']['n_frames'] * 2)
    model.to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'[info] Loaded model from {args.ckpt}')

    with torch.no_grad():
        y_true = []         # 1 for anomaly, 0 for normal
        y_pred = []         # prob of anomaly
        for step, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to(dtype=torch.float32, device=DEVICE)      # (N, C=20, H, W)

            # forward
            out = model(images)         # (N, 1), logits before sigmoid
            prob = torch.sigmoid(out.squeeze(-1))       # (N,)

            y_true.append(labels.to(dtype=torch.int32).cpu().numpy())
            y_pred.append(prob.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        try:
            auc = roc_auc_score(y_true, y_pred)
            acc = (y_true == (y_pred >= 0.5)).sum() / y_true.shape[0]
            print(f'[info] Video-Level AUC = {auc:.5f}, ACC = {acc*100:.2f}%')
        except:
            print('[warning] Failed to compute AUC and ACC.')

        try:
            with open(args.data_list, 'r') as fp:
                rows = [ln.rstrip().split() for ln in fp.readlines()]

            out_file = os.path.join(os.path.dirname(args.ckpt), 'results.npy')
            results = []
            for (name, label_str), target, prob in zip(rows, y_true, y_pred):
                assert (label_str == 'anomaly') == target
                results.append([name, label_str, target, prob])
            np.save(out_file, results)
        except:
            print('[warning] Failed to save output file.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt',
        default='',
        type=str,
        help='path to the model checkpoint'
    )
    parser.add_argument(
        '--data_root',
        default='/home/ubuntu/data_flow',
        type=str,
        help='root directory of optical flow images'
    )
    parser.add_argument(
        '--data_list',
        default='dataset/test.txt',
        type=str,
        help='path to the list of test videos'
    )
    parser.add_argument(
        '--image_size',
        default=224,
        type=int,
        help='height and width of the input image (default 224 for ResNets)'
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='batch size for test'
    )
    parser.add_argument(
        '--n_workers',
        default=4,
        type=int,
        help='number of workers for dataloader'
    )
    args = parser.parse_args()
    print(args)

    main(args)
