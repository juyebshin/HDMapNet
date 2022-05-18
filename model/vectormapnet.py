import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from .hdmapnet import HDMapNet
from .utils import onehot_encoding

def softargmax2d(input: Tensor, beta=100):
    *_, h, w = input.shape # b, c, h, w

    input = input.reshape(*_, h * w) # b, c, h*w
    input = F.softmax(beta * input, dim=-1) # b, c, h*w softmax over h*w dim

    indices_c, indices_r = np.meshgrid( # column, row
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    ) # (h, w) col, row mesh grid

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)), device=input.device) # 1, h*w
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)), device=input.device) # 1, h*w

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1) # b, c
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1) # b, c

    result = torch.stack([result_r, result_c], dim=-1)

    return result # b, c, 2

def MLP(channels: list, do_bn=True):
    """ MLP """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)
        )
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)

def remove_borders(vertices, scores, border: int, height: int, width: int):
        """ Removes vertices too close to the border """
        mask_h = (vertices[:, 0] >= border) & (vertices[:, 0] < (height - border))
        mask_w = (vertices[:, 1] >= border) & (vertices[:, 1] < (width - border))
        mask = mask_h & mask_w
        return vertices[mask], scores[mask]

def sample_dt(vertices, distance: Tensor, threshold: int, s: int = 8):
    """ Extract distance transform patches around vertices """
    # vertices: # tuple of length b, [N, 2(row, col)] tensor, in (25, 50) cell
    # distance: (b, 3, 200, 400) tensor
    embedding, _ = distance.max(1, keepdim=False) # b, 200, 400
    embedding = embedding / threshold # 0 ~ 10 -> 0 ~ 1 normalize
    b, h, w = embedding.shape # b, 200, 400
    hc, wc = int(h/s), int(w/s) # 25, 50
    embedding = embedding.reshape(b, hc, s, wc, s).permute(0, 1, 3, 2, 4) # b, 25, 8, 50, 8 -> b, 25, 50, 8, 8
    embedding = embedding.reshape(b, hc, wc, s*s) # b, 25, 50, 64
    embedding = [e[tuple(vc.t())] for e, vc in zip(embedding, vertices)] # tuple of length b, [N, 64] tensor
    return embedding

def normalize_vertices(vertices: torch.Tensor, image_shape):
    """ Normalize vertices locations in BEV space """
    # vertices: [N, 2] tensor in (x, y): (0~399, 0~199)
    _, height, width = image_shape # b, h, w
    one = vertices.new_tensor(1) # 1
    size = torch.stack([one*width, one*height])[None] # 1, 2
    center = size / 2 # 1, 2
    return (vertices - center) / center # N, 2

def top_k_vertices(vertices: torch.Tensor, scores: torch.Tensor, k: int):
    """Returns top-K vertices.

    vertices: [N, 2] tensor (N vertices in xy)
    scores: [N] tensor (N vertex scores)
    """
    # k: 300
    n_vertices = len(vertices) # N
    if k >= n_vertices:
        pad_size = k - n_vertices # k - N
        pad_v = torch.zeros([pad_size, 2])
        pad_s = torch.zeros([pad_size])
        vertices, scores = torch.cat([vertices, pad_v], dim=0), torch.cat([scores, pad_s], dim=0)
        mask = torch.zeros([k], dtype=torch.uint8)
        mask[:n_vertices] = 1
        return vertices, scores, mask # [K, 2], [K], [K]
    scores, indices = torch.topk(scores, k, dim=0)
    mask = torch.ones([k]) # [K]
    return vertices[indices], scores, mask # [K, 2], [K], [K]

class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=1):
        idx = torch.argmax(input, dim, keepdim=True) # b, 1, h, w

        output = torch.zeros_like(input) # b, c, h, w
        output.scatter_(dim, idx, 1) # b, c, h, w

        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class VectorMapNet(nn.Module):
    def __init__(self, data_conf, instance_seg=False, embedded_dim=16, direction_pred=False, direction_dim=36, lidar=False, distance_reg=True, vertex_pred=True, max_vertices=300) -> None:
        super(VectorMapNet, self).__init__()

        self.cell_size = data_conf['cell_size']
        self.dist_threshold = data_conf['dist_threshold']
        self.xbound = data_conf['xbound'][:-1] # [-30.0, 30.0]
        self.ybound = data_conf['ybound'][:-1] # [-15.0, 15.0]
        self.resolution = data_conf['xbound'][-1] # 0.15
        self.max_vertices = max_vertices

        self.center = torch.tensor([self.xbound[0], self.ybound[0]]).cuda() # -30.0, -15.0
        self.argmax = ArgMax.apply
        self.hdmapnet = HDMapNet(data_conf, instance_seg, embedded_dim, direction_pred, direction_dim, lidar, distance_reg, vertex_pred)

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        
        semantic, distance, vertex, embedding, direction = self.hdmapnet(img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll)
        # semantic, embedding, direction are not used

        # vertex: (b, 65, 25, 50)
        # distance: (b, 3, 200, 500)

        # Compute the dense vertices scores (heatmap)
        scores = F.softmax(vertex, 1) # (b, 65, 25, 50)
        # onehot = onehot_encoding(scores)[:, :-1] # b, 64, 25, 50, onehot over axis 64
        onehot = self.argmax(scores) # b, 65, 25, 50, onehot over axis 64
        onehot_nodust = onehot[:, :-1] # b, 64, 25, 50
        onehot_max, _ = onehot_nodust.max(1) # b, 25, 50
        vertices_cell = [torch.nonzero(vc) for vc in onehot_max] # list of length b, [N, 2(row, col)] tensor
        scores = scores[:, :-1] # b, 64, 25, 50
        b, _, h, w = scores.shape # b, 64, 25, 50
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.cell_size, self.cell_size) # b, 25, 50, 64 -> b, 25, 50, 8, 8
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*self.cell_size, w*self.cell_size) # b, 25, 8, 50, 8 -> b, 200, 400
        onehot_nodust = onehot_nodust.permute(0, 2, 3, 1).reshape(b, h, w, self.cell_size, self.cell_size) # b, 25, 50, 64 -> b, 25, 50, 8, 8
        onehot_nodust = onehot_nodust.permute(0, 1, 3, 2, 4).reshape(b, h*self.cell_size, w*self.cell_size) # b, 25, 8, 50, 8 -> b, 200, 400

        # Extract vertices
        vertices = [torch.nonzero(v) for v in onehot_nodust] # list of length b, [N, 2(row, col)] tensor
        scores = [s[tuple(v.t())] for s, v in zip(scores, vertices)] # list of length b, [N] tensor

        # Discard vertices near the image borders
        vertices, scores = list(zip(*[
            remove_borders(v, s, self.cell_size, h*self.cell_size, w*self.cell_size)
            for v, s in zip(vertices, scores)
        ])) # tuple

        if self.max_vertices >= 0:
            vertices, scores, masks = list(zip(*[
                top_k_vertices(v, s, self.max_vertices)
                for v, s in zip(vertices, scores)
            ]))

        # Convert (h, w) to (x, y), normalized
        # v: [N, 2]
        vertices = [normalize_vertices(torch.flip(v, [1]).float(), onehot_nodust.shape) for v in vertices] # list of [N, 2] tensor

        # Positional embedding (x, y, c)
        pos_embedding = [torch.cat((v, s.unsqueeze(1)), 1) for v, s in zip(vertices, scores)] # list of [N, 3] tensor
        pos_embedding = torch.stack(pos_embedding) # b, N, 3

        # Extract distance transform
        dt_embedding = sample_dt(vertices_cell, distance, self.dist_threshold, self.cell_size) # list of [N, 64] tensor
        dt_embedding = torch.stack(dt_embedding) # b, N, 64

        # vertices: N, 2 in XY vehicle space
        # scores: N vertex confidences

        return semantic, distance, vertex, embedding, direction
