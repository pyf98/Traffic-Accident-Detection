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
from networks import CRNNOpticalFlow


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    """Test the CRNN model on optical flow maps and save results as a numpy file.
    """
    ckpt = torch.load(args.ckpt)

    transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
    ])
    dataset = OpticalFlowDataset(
        root=args.data_root,
        video_list_path=args.data_list,
        n_frames=args.n_frames if args.n_frames > 0 else ckpt['net_configs']['n_frames'],
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

    model = CRNNOpticalFlow(
        cnn_dropout=ckpt['net_configs']['cnn_dropout'],
        cnn_emb_dim=ckpt['net_configs']['cnn_emb_dim'],
        cnn_type=ckpt['net_configs']['cnn_type'],
        rnn_hidden_size=ckpt['net_configs']['rnn_hidden_size'],
        rnn_dropout=ckpt['net_configs']['rnn_dropout'],
        num_rnn_layers=ckpt['net_configs']['num_rnn_layers'],
        rnn_bidir=ckpt['net_configs']['rnn_bidir']
    )
    model.to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'[info] Loaded model from {args.ckpt}')

    with torch.no_grad():
        y_true = []  # 1 for anomaly, 0 for normal
        y_pred = []  # prob of anomaly
        for step, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to(dtype=torch.float32, device=DEVICE)         # (N, T * 2, H, W)

            # forward
            out = model(images)                   # (N, T), probs after sigmoid
            out = torch.mean(out, dim=-1)         # (N,)

            y_true.append(labels.to(dtype=torch.int32).cpu().numpy())
            y_pred.append(out.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        try:
            auc = roc_auc_score(y_true, y_pred)
            acc = (y_true == (y_pred >= 0.5)).sum() / y_true.shape[0]
            print(f'[info] Video-Level AUC = {auc:.5f}, ACC = {acc*100:.3f}%')
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
        help='root directory of optical flow maps'
    )
    parser.add_argument(
        '--data_list',
        default='dataset/test.txt',
        type=str,
        help='path to the list of test videos'
    )
    parser.add_argument(
        '--n_frames',
        default=0,
        type=int,
        help='number of frames for each video clip'
    )
    parser.add_argument(
        '--image_size',
        default=224,
        type=int,
        help='height and width of the input image (default 224 for ResNets)'
    )
    parser.add_argument(
        '--batch_size',
        default=8,
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
