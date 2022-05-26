from turtle import forward
from copy import deepcopy
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

def normalize_vertices(vertices: Tensor, image_shape):
    """ Normalize vertices locations in BEV space """
    # vertices: [N, 2] tensor in (x, y): (0~399, 0~199)
    _, height, width = image_shape # b, h, w
    one = vertices.new_tensor(1) # [1], data 1
    size = torch.stack([one*width, one*height])[None] # [1, 2], data [400, 200]
    center = size / 2 # [1, 2], data [200, 100]
    return (vertices - center) / size # [N, 2] values [-0.5, 0.5]

def top_k_vertices(vertices: Tensor, scores: Tensor, embeddings: Tensor, k: int):
    """
    Returns top-K vertices.

    vertices: [N, 2] tensor (N vertices in xy)
    scores: [N] tensor (N vertex scores)
    embeddings: [N, 64] tensor
    """
    # k: 300
    n_vertices = len(vertices) # N
    embedding_dim = embeddings.shape[1]
    if k >= n_vertices:
        pad_size = k - n_vertices # k - N
        pad_v = torch.zeros([pad_size, 2], device=vertices.device)
        pad_s = torch.zeros([pad_size], device=scores.device)
        pad_dt = torch.zeros([pad_size, embedding_dim], device=embeddings.device)
        vertices, scores, embeddings = torch.cat([vertices, pad_v], dim=0), torch.cat([scores, pad_s], dim=0), torch.cat([embeddings, pad_dt], dim=0)
        mask = torch.zeros([k], dtype=torch.uint8, device=vertices.device)
        mask[:n_vertices] = 1
        return vertices, scores, embeddings, mask # [K, 2], [K], [K]
    scores, indices = torch.topk(scores, k, dim=0)
    mask = torch.ones([k], dtype=torch.uint8, device=vertices.device) # [K]
    return vertices[indices], scores, embeddings[indices], mask # [K, 2], [K], [K]

def attention(query, key, value, mask=None):
    # q, k, v: [b, 64, 4, N], mask: [b, 1, N]
    dim = query.shape[1] # 64
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5 # [b, 4, N, N], dim**.5 == 8
    if mask is not None:
        mask = torch.einsum('bdn,bdm->bdnm', mask, mask) # [b, 1, N, N]
        scores = scores.masked_fill(mask == 0, -1e9)
    prob = torch.nn.functional.softmax(scores, dim=-1) # [b, 4, N, N]
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob # final message passing [b, 64, 4, N]



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

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads # 64
        self.num_heads = num_heads # 4
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, mask=None):
        # q, k, v: [b, 256, N], mask: [b, 1, N]
        batch_dim = query.size(0) # b
        num_vertices = query.size(2) # N
        if mask is None:
            mask = torch.ones([batch_dim, 1, num_vertices], device=query.device) # [b, 1, N]
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        # q, k, v: [b, 64, 4, N]
        x, _ = attention(query, key, value, mask) # [b, 64, 4, N]
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, mask=None):
        # attn(q, k, v)
        message = self.attn(x, source, source, mask) # [b, 256, N]
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, embedding, mask=None):
        # Only self-attention is implemented for now
        # embedding: [b, 256, N]
        # mask: [b, 1, N]
        for layer, name in zip(self.layers, self.names):
            # if name == 'cross':
            #     src0, src1 = desc1, desc0
            # else:  # if name == 'self':
            #     src0, src1 = desc0, desc1
            delta = layer(embedding, embedding, mask) # [b, 256, N]
            embedding = (embedding + delta) # [b, 256, N]
        return embedding

class GraphEncoder(nn.Module):
    """ Joint encoding of vertices and distance transform embeddings """
    def __init__(self, feature_dim, layers: list) -> None:
        super().__init__()
        # first element of layers should be either 3 (for vertices) or 64 (for dt embeddings)
        self.encoder = MLP(layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)
    
    def forward(self, embedding: torch.Tensor):
        """ vertices: [b, N, 3] vertices coordinates with score confidence (x y c)
            distance: [b, N, 64]
        """
        input = embedding.transpose(1, 2) # [b, C, N] C = 3 for vertices, C = 64 for dt
        return self.encoder(input) # [b, 256, N]

class VectorMapNet(nn.Module):
    def __init__(self, data_conf, instance_seg=False, embedded_dim=16, direction_pred=False, direction_dim=36, lidar=False, distance_reg=True) -> None:
        super(VectorMapNet, self).__init__()

        self.cell_size = data_conf['cell_size']
        self.dist_threshold = data_conf['dist_threshold']
        self.xbound = data_conf['xbound'][:-1] # [-30.0, 30.0]
        self.ybound = data_conf['ybound'][:-1] # [-15.0, 15.0]
        self.resolution = data_conf['xbound'][-1] # 0.15
        self.vertex_threshold = data_conf['vertex_threshold'] # 0.015
        self.max_vertices = data_conf['num_vectors']*3 # 100*3
        self.feature_dim = data_conf['feature_dim'] # 256
        # self.GNN_layers = gnn_layers

        self.center = torch.tensor([self.xbound[0], self.ybound[0]]).cuda() # -30.0, -15.0
        self.argmax = ArgMax.apply

        # Intermediate representations: vertices, distance transform
        self.bev_backbone = HDMapNet(data_conf, False, instance_seg, embedded_dim, direction_pred, direction_dim, lidar, distance_reg, vertex_pred=True)

        # Graph neural network
        self.venc = GraphEncoder(self.feature_dim, [3, 32, 64, 128, 256]) # 3 -> 256
        self.dtenc = GraphEncoder(self.feature_dim, [self.cell_size*self.cell_size, 64, 128, 256]) # 64 -> 256
        self.gnn = AttentionalGNN(self.feature_dim, data_conf['gnn_layers'])
        self.final_proj = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1, bias=True)

        # bin_score = nn.Parameter(torch.tensor(1.))
        # self.register_parameter('bin_score', bin_score)

        self.matching = nn.Sigmoid()

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        """ semantic, embedding, direction are not used

        @ vertex: (b, 65, 25, 50)
        @ distance: (b, 3, 200, 500)
        """
        
        semantic, distance, vertex, embedding, direction = self.bev_backbone(img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll)

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

        # scores = scores[:, :-1].permute(0, 2, 3, 1) # b, 25, 50, 64
        # scores[scores < self.vertex_threshold] = 0.0
        # scores_max, max_idx = scores.max(-1) # b, 25, 50, 1
        # vertices_cell = [torch.nonzero(vc.squeeze(-1)) for vc in scores_max] # list of length b, [N, 2(row, col)] tensor, (row, col) within (25, 50)

        # Extract vertices
        # vertices: [N, 2] in XY vehicle space
        # scores: [N] vertex confidences
        vertices = [torch.nonzero(v) for v in onehot_nodust] # list of length b, [N, 2(row, col)] tensor
        scores = [s[tuple(v.t())] for s, v in zip(scores, vertices)] # list of length b, [N] tensor
        # Extract distance transform
        dt_embedding = sample_dt(vertices_cell, distance, self.dist_threshold, self.cell_size) # list of [N, 64] tensor

        # Discard vertices near the image borders
        # vertices, scores = list(zip(*[
        #     remove_borders(v, s, self.cell_size, h*self.cell_size, w*self.cell_size)
        #     for v, s in zip(vertices, scores)
        # ])) # tuple

        if self.max_vertices >= 0:
            vertices, scores, dt_embedding, masks = list(zip(*[
                top_k_vertices(v, s, d, self.max_vertices)
                for v, s, d in zip(vertices, scores, dt_embedding)
            ]))

        # Convert (h, w) to (x, y), normalized
        # v: [N, 2]
        vertices = [normalize_vertices(torch.flip(v, [1]).float(), onehot_nodust.shape) for v in vertices] # list of [N, 2] tensor

        # Positional embedding (x, y, c)
        pos_embedding = [torch.cat((v, s.unsqueeze(1)), 1) for v, s in zip(vertices, scores)] # list of [N, 3] tensor
        pos_embedding = torch.stack(pos_embedding) # [b, N, 3]

        dt_embedding = torch.stack(dt_embedding) # [b, N, 64]
        masks = torch.stack(masks).unsqueeze(-1) # [b, N, 1]

        graph_embedding = self.venc(pos_embedding) + self.dtenc(dt_embedding) # [b, 256, N]
        # masks = masks.transpose(1, 2) # [b, 1, N]
        graph_embedding = self.gnn(graph_embedding, masks.transpose(1, 2)) # [b, 256, N]
        graph_embedding = self.final_proj(graph_embedding) # [b, 256, N]

        # Adjacency matrix score as inner product of all nodes
        scores = torch.einsum('bdn,bdm->bnm', graph_embedding, graph_embedding)
        scores = scores / self.feature_dim**.5 # [b, N, N]

        """ Matching layer (put these in a function or class) """
        # b, m, n = scores.shape
        # one = scores.new_tensor(1)
        # ms, ns = (m*one).to(scores), (n*one).to(scores) # tensor(N), tensor(N)

        # # alpha = self.bin_score
        # bins0 = self.bin_score.expand(b, m, 1) # [b, N, 1]
        # bins1 = self.bin_score.expand(b, 1, n) # [b, 1, N]
        # alpha = self.bin_score.expand(b, 1, 1) # [b, 1, 1]

        # couplings = torch.cat( # [b, N+1, N+1]
        #     [
        #         torch.cat([scores, bins0], -1), # [b, N, N+1]
        #         torch.cat([bins1, alpha], -1)   # [b, 1, N+1]
        #     ], 1)

        # Symmetry property
        # scores = (scores.transpose(1, 2) + scores) * 0.5

        # return scores [b, N, N], pos_embedding (normalized -0.5~0.5 with scores) [b, N, 3], masks [b, N, 1]

        return semantic, distance, vertex, embedding, direction, self.matching(scores), pos_embedding, masks
