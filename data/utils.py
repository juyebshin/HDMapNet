import numpy as np

import torch
import cv2


def get_proj_mat(intrins, rots, trans):
    K = np.eye(4)
    K[:3, :3] = intrins
    R = np.eye(4)
    R[:3, :3] = rots.transpose(-1, -2)
    T = np.eye(4)
    T[:3, 3] = -trans
    RT = R @ T
    return K @ RT


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def label_onehot_decoding(onehot):
    return torch.argmax(onehot, axis=0)


def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1)
    return onehot


def gen_dx_bx(xbound, ybound, zbound):
    # xbound: [-30.0, 30.0, 0.15], ybound: [-15.0, 15.0, 0.15], zbound: [-10.0, 10.0, 20.0]
    # dx = tensor([ 0.1500,  0.1500, 20.0000])
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    # bx = tensor([-29.9250, -14.9250,   0.0000])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    # nx = tensor([400, 200,   1])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def get_distance_transform(masks, threshold=None):
    # masks: (3, 196, 200) np bool
    labels = (~masks).astype('uint8')
    distances = np.zeros(masks.shape, dtype=np.float32)
    for i, label in enumerate(labels):
        distances[i] = cv2.distanceTransform(label, cv2.DIST_L2, maskSize=5)
        # truncate to [0.0, 10.0] and invert values
        if threshold is not None:
            distances[i] = float(threshold) - distances[i]
            distances[i][distances[i] < 0.0] = 0.0
        # cv2.normalize(distances[i], distances[i], 0, 1.0, cv2.NORM_MINMAX)
    return distances