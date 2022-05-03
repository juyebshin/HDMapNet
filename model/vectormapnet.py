import torch
from torch import nn
import torch.nn.functional as F

from .hdmapnet import HDMapNet
from evaluate import onehot_encoding

class VectorMapNet(nn.Module):
    def __init__(self, data_conf, instance_seg=False, embedded_dim=16, direction_pred=False, direction_dim=36, lidar=False, distance_reg=True, vertex_pred=True) -> None:
        super(VectorMapNet, self).__init__()

        self.cell_size = data_conf['cell_size']
        self.xbound = data_conf['xbound'][:-1] # [-30.0, 30.0]
        self.ybound = data_conf['ybound'][:-1] # [-15.0, 15.0]
        self.resolution = data_conf['xbound'][-1] # 0.15

        self.hdmapnet = HDMapNet(data_conf, instance_seg, embedded_dim, direction_pred, direction_dim, lidar, distance_reg, vertex_pred)

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        semantic, distance, vertex, embedding, direction = self.hdmapnet(img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll)
        # semantic, embedding, direction are not used

        # vertex: (b, 65, 25, 50)

        # Compute the dense vertices scores (heatmap)
        scores = F.softmax(vertex, 1) # (b, 65, 25, 50)
        onehot = onehot_encoding(scores)[:, :-1] # b, 64, 25, 50, onehot over axis 64
        scores = scores[:, :-1] # b, 64, 25, 50
        b, _, h, w = scores.shape # b, 64, 25, 50
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.cell_size, self.cell_size) # b, 25, 50, 64 -> b, 25, 50, 8, 8
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*self.cell_size, w*self.cell_size) # b, 25, 8, 50, 8 -> b, 200, 400
        onehot = onehot.permute(0, 2, 3, 1).reshape(b, h, w, self.cell_size, self.cell_size) # b, 25, 50, 64 -> b, 25, 50, 8, 8
        onehot = onehot.permute(0, 1, 3, 2, 4).reshape(b, h*self.cell_size, w*self.cell_size) # b, 25, 8, 50, 8 -> b, 200, 400

        # Extract vertices
        vertices = [torch.nonzero(s) for s in onehot] # tuple of length b, [N, 2(row, col)] tensor
        scores = [s[tuple(v.t())] for s, v in zip(scores, vertices)] # tuple of length b, [N] tensor

        center = torch.tensor([self.xbound[0], self.ybound[0]]) # -30.0, -15.0
        # Convert (h, w) to (x, y) h: 0~199, w: 0~399 -> x: -30~30, y: -15~15
        # v: [N, 2]
        vertices = [torch.flip(v, [1]).float().mul(self.resolution).add(center) for v in vertices]

        # vertices: N, 2 in XY vehicle space
        # scores: N vertex confidences

        return semantic, distance, vertex, embedding, direction