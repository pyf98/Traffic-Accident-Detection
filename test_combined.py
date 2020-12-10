import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score


def main(args):
    """Combine video-level predictions from two streams using weighted average fusion.
    Notes:
        combined_prob = rgb_weight * rgb_prob + (1. - rgb_weight) * flow_prob
    """
    rgb_results = np.load(args.rgb_file)
    flow_results = np.load(args.flow_file)
    rgb_w = float(args.rgb_weight)
    assert 0. <= rgb_w <= 1.

    y_true = []
    y_pred_rgb = []
    y_pred_flow = []
    for (name1, label_str1, target1, prob1), (name2, label_str2, target2, prob2) in zip(rgb_results, flow_results):
        assert name1 == name2
        assert label_str1 == label_str2
        assert target1 == target2

        y_true.append(int(target1))
        y_pred_rgb.append(float(prob1))
        y_pred_flow.append(float(prob2))

    y_true = np.array(y_true, dtype=np.int)
    y_pred_rgb = np.array(y_pred_rgb, dtype=np.float)
    y_pred_flow = np.array(y_pred_flow, dtype=np.float)
    y_pred = y_pred_rgb * rgb_w + y_pred_flow * (1. - rgb_w)

    auc_rgb = roc_auc_score(y_true, y_pred_rgb)
    auc_flow = roc_auc_score(y_true, y_pred_flow)
    auc = roc_auc_score(y_true, y_pred)

    acc_rgb = ((y_pred_rgb >= 0.5) == y_true).sum() / y_true.shape[0]
    acc_flow = ((y_pred_flow >= 0.5) == y_true).sum() / y_true.shape[0]
    acc = ((y_pred >= 0.5) == y_true).sum() / y_true.shape[0]

    print(f'=============== AUC ===============')
    print(f'== RGB:  {auc_rgb:.5f}')
    print(f'== Flow: {auc_flow:.5f}')
    print(f'== Both: {auc:.5f}')
    print(f'============= Accuracy ==============')
    print(f'== RGB:  {acc_rgb*100:.3f}%')
    print(f'== Flow: {acc_flow*100:.3f}%')
    print(f'== Both: {acc*100:.3f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rgb_file',
        type=str,
        help='path to the npy file of RGB frames'
    )
    parser.add_argument(
        '--flow_file',
        type=str,
        help='path to the npy file of optical flow images'
    )
    parser.add_argument(
        '--rgb_weight',
        default=0.5,
        type=float,
        help='weight of RGB predictions (0.0 to 1.0)'
    )
    args = parser.parse_args()
    print(args)

    main(args)
