import matplotlib
matplotlib.use('Agg')   # Don't show figure 
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, precision_recall_curve

from network.ProtoAD import ProtoAD
from network.utils import select_device, set_seed
from network.mvtec import MVTecDataset, denormalization

def parse_args():
    """ Argument Parser """
    parser = argparse.ArgumentParser("ProtoAD")
    
    parser.add_argument('--class_name', type=str, default='carpet')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--cropsize', type=int, default=224)

    parser.add_argument('--ckpt_dir', type=str, default='./prototype/bottle_prototypes.pt')
    parser.add_argument('--seed', type=int, default=42) 
    
    return parser.parse_args()


feature_dim = 256+512+1024
output_stride = 4


def main(args):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    model = ProtoAD()
    model.to(args.device)

    prototype_path = os.path.join(args.ckpt_dir, '%s_prototypes.pt' % args.class_name)
    print('load prototypes from disk.')
    prototype = torch.load(prototype_path)

    print("=== Class(%s), Cluster Num(%d) ===" % (args.class_name, prototype.shape[0]))    
    model.update_prototype(prototype.to(args.device))
    test_dataset = MVTecDataset(class_name=args.class_name, is_train=False, resize=args.resize, cropsize=args.cropsize)   # MVTec
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # testing
    model.eval()
    img_score_list, score_map_list, gt_label_list, gt_mask_list = [], [], [], []
    
    for images, label, mask, _ in tqdm(test_dataloader):
        with torch.no_grad():
            anomaly_map = model.anomaly_forward(images.to(args.device))    # [B,C,H/8,W/8]
        # pixel-level score map
        score_map = anomaly_map.squeeze().detach().cpu().numpy()
        score_map = cv2.resize(score_map, (args.cropsize, args.cropsize))
        score_map = gaussian_filter(score_map, sigma=4)
        # image-level score
        img_score = score_map.max()

        img_score_list.append(img_score)
        score_map_list.append(score_map)
        gt_label_list.append(label.detach().cpu().numpy())
        gt_mask_list.append(mask.squeeze().detach().cpu().numpy())

    # Evaluation
    img_auc = roc_auc_score(np.asarray(gt_label_list).flatten(), np.asarray(img_score_list).flatten())
    print("Image-Level AUCROC : %.1f" % (img_auc*100))
    pixel_auc = roc_auc_score(np.asarray(gt_mask_list).flatten(), np.asarray(score_map_list).flatten())
    print("Pixel-Level AUCROC : %.1f" % (pixel_auc*100))


if __name__ == '__main__':    
    args = parse_args()
    device = select_device(args.gpu_id)
    args.device = device
    set_seed(args.seed)
    main(args)