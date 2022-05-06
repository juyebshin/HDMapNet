import torch
from torch import nn
import torch.nn.functional as F

from .hdmapnet import HDMapNet
from .utils import onehot_encoding

class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=1):
        idx = torch.argmax(input, dim, keepdim=True) # b, 1, h, w

        output = torch.zeros_like(input)
        output.scatter_(dim, idx, 1)

        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class VectorMapNet(nn.Module):
    def __init__(self, data_conf, instance_seg=False, embedded_dim=16, direction_pred=False, direction_dim=36, lidar=False, distance_reg=True, vertex_pred=True) -> None:
        super(VectorMapNet, self).__init__()

        self.cell_size = data_conf['cell_size']
        self.dist_threshold = data_conf['dist_threshold']
        self.xbound = data_conf['xbound'][:-1] # [-30.0, 30.0]
        self.ybound = data_conf['ybound'][:-1] # [-15.0, 15.0]
        self.resolution = data_conf['xbound'][-1] # 0.15

        self.argmax = ArgMax.apply
        self.hdmapnet = HDMapNet(data_conf, instance_seg, embedded_dim, direction_pred, direction_dim, lidar, distance_reg, vertex_pred)

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        semantic, distance, vertex, embedding, direction = self.hdmapnet(img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll)
        # semantic, embedding, direction are not used

        # vertex: (b, 65, 25, 50)
        # distance: (b, 3, 200, 500)

        # Compute the dense vertices scores (heatmap)
        scores = F.softmax(vertex, 1) # (b, 65, 25, 50)
        onehot = onehot_encoding(scores)[:, :-1] # b, 64, 25, 50, onehot over axis 64
        onehot_max, _ = onehot.max(1) # b, 25, 50
        vertices_cell = [torch.nonzero(vc) for vc in onehot_max] # tuple of length b, [N, 2(row, col)] tensor
        scores = scores[:, :-1] # b, 64, 25, 50
        b, _, h, w = scores.shape # b, 64, 25, 50
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.cell_size, self.cell_size) # b, 25, 50, 64 -> b, 25, 50, 8, 8
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*self.cell_size, w*self.cell_size) # b, 25, 8, 50, 8 -> b, 200, 400
        onehot = onehot.permute(0, 2, 3, 1).reshape(b, h, w, self.cell_size, self.cell_size) # b, 25, 50, 64 -> b, 25, 50, 8, 8
        onehot = onehot.permute(0, 1, 3, 2, 4).reshape(b, h*self.cell_size, w*self.cell_size) # b, 25, 8, 50, 8 -> b, 200, 400

        # Extract vertices
        vertices = [torch.nonzero(v) for v in onehot] # tuple of length b, [N, 2(row, col)] tensor
        scores = [s[tuple(v.t())] for s, v in zip(scores, vertices)] # tuple of length b, [N] tensor

        # Discard vertices near the image borders
        vertices, scores = list(zip(*[
            self.remove_borders(v, s, self.cell_size, h*self.cell_size, w*self.cell_size)
            for v, s in zip(vertices, scores)
        ]))

        center = torch.tensor([self.xbound[0], self.ybound[0]]).cuda() # -30.0, -15.0
        # Convert (h, w) to (x, y) h: 0~199, w: 0~399 -> x: -30~30, y: -15~15
        # v: [N, 2]
        vertices = [torch.flip(v, [1]).float().mul(self.resolution).add(center) for v in vertices]

        # Extract distance transform
        dt_embedding = self.sample_dt(vertices_cell, distance, self.cell_size)

        # vertices: N, 2 in XY vehicle space
        # scores: N vertex confidences

        return semantic, distance, vertex, embedding, direction

    def remove_borders(self, vertices, scores, border: int, height: int, width: int):
        """ Removes vertices too close to the border """
        mask_h = (vertices[:, 0] >= border) & (vertices[:, 0] < (height - border))
        mask_w = (vertices[:, 1] >= border) & (vertices[:, 1] < (width - border))
        mask = mask_h & mask_w
        return vertices[mask], scores[mask]

    def sample_dt(self, vertices, distance: torch.Tensor, s: int = 8):
        """ Extract distance transform patches around vertices """
        # vertices: # tuple of length b, [N, 2(row, col)] tensor, in (25, 50) cell
        # distance: (b, 3, 200, 400) tensor
        embedding, _ = distance.max(1, keepdim=False) # b, 200, 400
        embedding = embedding / self.dist_threshold # 0 ~ 10 -> 0 ~ 1 normalize
        b, h, w = embedding.shape # b, 200, 400
        hc, wc = int(h/s), int(w/s) # 25, 50
        embedding = embedding.reshape(b, hc, s, wc, s).permute(0, 1, 3, 2, 4) # b, 25, 8, 50, 8 -> b, 25, 50, 8, 8
        embedding = embedding.reshape(b, hc, wc, s*s) # b, 25, 50, 64
        embedding = [e[tuple(vc.t())] for e, vc in zip(embedding, vertices)] # tuple of length b, [N, 64] tensor
        return embedding