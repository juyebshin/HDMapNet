from tkinter.messagebox import NO
from turtle import forward
from copy import deepcopy
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from .hdmapnet import HDMapNet
from .gcn import GCN

def MLP(channels: list, do_bn=True, norm_layer=nn.BatchNorm1d):
    """ MLP """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)
        )
        if i < (n-1):
            if do_bn:
                layers.append(norm_layer(channels[i]))
            layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def remove_borders(vertices, scores, border: int, height: int, width: int):
        """ Removes vertices too close to the border """
        mask_h = (vertices[:, 0] >= border) & (vertices[:, 0] < (height - border))
        mask_w = (vertices[:, 1] >= border) & (vertices[:, 1] < (width - border))
        mask = mask_h & mask_w
        return vertices[mask], scores[mask]

def sample_dt(vertices, distance: Tensor, s: int = 8):
    """ Extract distance transform patches around vertices """
    # vertices: # tuple of length b, [N, 2(row, col)] tensor, in (25, 50) cell
    # distance: (b, 3, 200, 400) tensor
    # embedding, _ = distance.max(1, keepdim=False) # b, 200, 400
    embedding = distance # 0 ~ 10 -> 0 ~ 1 normalize
    b, c, h, w = embedding.shape # b, 3, 200, 400
    hc, wc = int(h/s), int(w/s) # 25, 50
    embedding = embedding.reshape(b, c, hc, s, wc, s).permute(0, 1, 2, 4, 3, 5) # b, c, 25, 8, 50, 8 -> b, c, 25, 50, 8, 8
    embedding = embedding.reshape(b, c, hc, wc, s*s).permute(0, 2, 3, 1, 4) # b, c, 25, 50, 64 -> b, 25, 50, 3, 64
    embedding = embedding.reshape(b, hc, wc, -1) # b, 25, 50, 192
    embedding = [e[tuple(vc.t())] for e, vc in zip(embedding, vertices)] # tuple of length b, [N, 192] tensor
    return embedding

def sample_feat(vertices, feature: Tensor):
    """ Extract feature patches around vertices """
    # vertices: # tuple of length b, [N, 2(row, col)] tensor, in (25, 50) cell
    # feature: (b, 256, 25, 50) tensor
    b, c, h, w = feature.shape # b, 256, 25, 50
    embedding = feature.permute(0, 2, 3, 1) # [b, 25, 50, 256]
    embedding = [e[tuple(vc.t())] for e, vc in zip(embedding, vertices)] # tuple of length b, [N, 256] tensor
    return embedding

def normalize_vertices(vertices: Tensor, image_shape):
    """ Normalize vertices locations in BEV space """
    # vertices: [N, 2] tensor in (x, y): (0~399, 0~199)
    _, height, width = image_shape # b, h, w
    one = vertices.new_tensor(1) # [1], values 1
    size = torch.stack([one*width, one*height])[None] # [1, 2], values [400, 200]
    center = size / 2 # [1, 2], values [200, 100]
    return (vertices - center + 0.5) / size # [N, 2] values [-0.5, 0.4975] or [-0.49875, 0.49875]

def top_k_vertices(vertices: Tensor, scores: Tensor, embeddings: Tensor, k: int):
    """
    Returns top-K vertices.

    vertices: [N, 2] tensor (N vertices in xy)
    scores: [N] tensor (N vertex scores)
    embeddings: [N, 64] tensor
    """
    # k: 400
    n_vertices = len(vertices) # N
    embedding_dim = embeddings.shape[1]
    if k >= n_vertices:
        pad_size = k - n_vertices # k - N
        pad_v = torch.ones([pad_size, 2], device=vertices.device, requires_grad=False)
        pad_s = torch.ones([pad_size], device=scores.device, requires_grad=False)
        pad_dt = torch.ones([pad_size, embedding_dim], device=embeddings.device, requires_grad=False)
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


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):    
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape # b, N, N
    # m_valid = n_valid = torch.count_nonzero(masks, 1).squeeze(-1) # [b] number of valid
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores) # [b] same as m_valid, n_valid

    bins0 = alpha.expand(b, m, 1) # [b, N, 1]
    bins1 = alpha.expand(b, 1, n) # [b, 1, N]
    alpha = alpha.expand(b, 1, 1) # [b, 1, 1]

    couplings = torch.cat( # [b, N+1, N+1]
        [
            torch.cat([scores, bins0], -1), # [b, N, N+1]
            torch.cat([bins1, alpha], -1)   # [b, 1, N+1]
        ], 1)
    # masks_bins = torch.cat([masks, masks.new_tensor(1).expand(b, 1, 1)], 1) # [b, N+1, 1]

    norm = - (ms + ns).log() # [b]
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm]) # [N+1]
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm]) # [N+1]
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1) # [b, N+1]

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

# Positional embedding from NeRF: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py
def get_embedder(multires, i=0):

    if i == -1:
        return torch.nn.Identity(), 2

    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

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
        # q, k, v: [b, dim, head, N]
        x, _ = attention(query, key, value, mask) # [b, 64, head, N], [b, head, N, N]
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim], norm_layer=norm_layer)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, mask=None):
        # x, source: [b, 256(feature_dim), N]
        # attn(q, k, v)
        message = self.attn(x, source, source, mask) # [b, 256, N], [b, 4, N, N]
        return self.mlp(torch.cat([x, message], dim=1)) # [4, 512, 300] -> [4, 256, 300], [b, 4, N, N]

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4, norm_layer)
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
            # Attentional propagation
            delta = layer(embedding, embedding, mask) # [b, 256, N], [b, 4, N, N]
            embedding = (embedding + delta) # [b, 256, N]
        return embedding

class GraphEncoder(nn.Module):
    """ Joint encoding of vertices and distance transform embeddings """
    def __init__(self, feature_dim, layers: list, norm_layer=nn.BatchNorm1d) -> None:
        super().__init__()
        # first element of layers should be either 3 (for vertices) or 64 (for dt embeddings)
        self.encoder = MLP(layers + [feature_dim], norm_layer=norm_layer)
        nn.init.constant_(self.encoder[-1].bias, 0.0)
    
    def forward(self, embedding: torch.Tensor):
        """ vertices: [b, N, 3] vertices coordinates with score confidence (x y c)
            distance: [b, N, 64]
        """
        input = embedding.transpose(1, 2) # [b, C, N] C = 3 for vertices, C = 64 for dt
        return self.encoder(input) # [b, 256, N]

class VectorMapNet(nn.Module):
    def __init__(self, data_conf, norm_layer_dict, instance_seg=False, embedded_dim=16, direction_pred=False, direction_dim=36, lidar=False, distance_reg=True, refine=False) -> None:
        super(VectorMapNet, self).__init__()

        self.num_classes = data_conf['num_channels'] # 4
        self.cell_size = data_conf['cell_size']
        self.dist_threshold = data_conf['dist_threshold']
        self.distance_reg = distance_reg
        self.xbound = data_conf['xbound'][:-1] # [-30.0, 30.0]
        self.ybound = data_conf['ybound'][:-1] # [-15.0, 15.0]
        self.resolution = data_conf['xbound'][-1] # 0.15
        self.vertex_threshold = data_conf['vertex_threshold'] # 0.015
        self.max_vertices = data_conf['num_vectors'] # 300
        self.feature_dim = data_conf['feature_dim'] # 256
        self.pos_freq = data_conf['pos_freq']
        self.sinkhorn_iters = data_conf['sinkhorn_iterations'] # 100 default 0: not using sinkhorn
        self.gnn_layers = data_conf['gnn_layers']
        self.refine = refine

        self.center = torch.tensor([self.xbound[0], self.ybound[0]]).cuda() # -30.0, -15.0

        # Intermediate representations: vertices, distance transform
        self.bev_backbone = HDMapNet(data_conf, norm_layer_dict['2d'], False, instance_seg, embedded_dim, direction_pred, direction_dim, lidar, distance_reg, vertex_pred=True)

        # Positional encoding
        self.pe_fn, self.pe_dim = get_embedder(data_conf['pos_freq'])
        
        # Graph neural network
        # self.pe_dim = self.pe_dim + 1 with confidence added, here 42+1
        self.venc = GraphEncoder(self.feature_dim, [self.pe_dim + 1, 64, 128, 256], norm_layer_dict['1d']) # 43 -> 64 -> 128 -> 256 -> 256
        embedding_dim = (self.num_classes-1)*self.cell_size*self.cell_size if distance_reg else 256 # 192 or 256
        self.dtenc = GraphEncoder(self.feature_dim, [embedding_dim, 64, 128, 256], norm_layer_dict['1d']) # 192/256 -> 128 -> 256 for visual descriptor
        # if distance_reg:
        #     self.dtenc = GraphEncoder(self.feature_dim, [embedding_dim, 64, 128, 256], norm_layer_dict['1d']) # 192/256 -> 128 -> 256
            # self.dtenc = GraphEncoder(self.feature_dim, [embedding_dim+3, 64, 128, 256], norm_layer_dict['1d']) # temp
        self.gnn = AttentionalGNN(self.feature_dim, ['self']*self.gnn_layers, norm_layer_dict['1d'])
        self.final_proj = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1, bias=True)

        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        # self.gcn = GCN(self.feature_dim, 512, self.num_classes, 0.5)
        self.cls_head = nn.Conv1d(self.feature_dim, self.num_classes-1, kernel_size=1, bias=True)
        if self.refine:
            self.offset_head = nn.Conv1d(self.feature_dim, 2, kernel_size=1, bias=True)

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        """ semantic, instance, direction are not used
        @ vertex: (b, 65, 25, 50)
        @ distance: (b, 3, 200, 400)
        """
        
        semantic, distance, vertex, instance, direction = self.bev_backbone(img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll)

        # Compute the dense vertices scores (heatmap)
        scores = F.softmax(vertex, 1) # (b, 65, 25, 50)
        scores = scores[:, :-1] # b, 64, 25, 50
        b, _, h, w = scores.shape # b, 64, 25, 50
        mvalues, mindicies = scores.max(1, keepdim=True) # b, 1, 25, 50
        scores_max = scores.new_full(scores.shape, 0., dtype=scores.dtype)
        scores_max = scores_max.scatter_(1, mindicies, mvalues) # b, 64, 25, 50
        scores_max = scores_max.permute(0, 2, 3, 1).reshape(b, h, w, self.cell_size, self.cell_size) # b, 25, 50, 64 -> b, 25, 50, 8, 8
        scores_max = scores_max.permute(0, 1, 3, 2, 4).reshape(b, h*self.cell_size, w*self.cell_size) # b, 25, 8, 50, 8 -> b, 200, 400
        scores_max = simple_nms(scores_max, int(self.cell_size*0.5)) # b, 200, 400
        score_shape = scores_max.shape # b, 200, 400

        # scores = scores[:, :-1].permute(0, 2, 3, 1) # b, 25, 50, 64
        # scores[scores < self.vertex_threshold] = 0.0
        # scores_max, max_idx = scores.max(-1) # b, 25, 50, 1
        # vertices_cell = [torch.nonzero(vc.squeeze(-1)) for vc in scores_max] # list of length b, [N, 2(row, col)] tensor, (row, col) within (25, 50)

        # [1] Extract vertices
        # onehot_nodust = onehot[:, :-1] # b, 64, 25, 50
        # onehot_max, _ = onehot_nodust.max(1) # b, 25, 50
        # vertices_cell = [torch.nonzero(vc) for vc in onehot_max] #
        # onehot_nodust = onehot_nodust.permute(0, 2, 3, 1).reshape(b, h, w, self.cell_size, self.cell_size) # b, 25, 50, 64 -> b, 25, 50, 8, 8
        # onehot_nodust = onehot_nodust.permute(0, 1, 3, 2, 4).reshape(b, h*self.cell_size, w*self.cell_size) # b, 25, 8, 50, 8 -> b, 200, 400
        # vertices: [N, 2] in XY vehicle space
        # scores: [N] vertex confidences
        # vertices = [torch.nonzero(v) for v in onehot_nodust] # list of length b, [N, 2(row, col)] tensor
        # scores = [s[tuple(v.t())] for s, v in zip(scores, vertices)] # list of length b, [N] tensor

        # [2] Extract vertices using NMS
        vertices = [torch.nonzero(s > self.vertex_threshold) for s in scores_max] # list of length b, [N, 2(row, col)] tensor
        scores = [s[tuple(v.t())] for s, v in zip(scores_max, vertices)] # list of length b, [N] tensor
        vertices_cell = [(v / self.cell_size).trunc().long() for v in vertices]

        # Extract distance transform
        if self.distance_reg:
            dt_embedding = sample_dt(vertices_cell, F.relu(distance).clamp(max=self.dist_threshold), self.cell_size) # list of [N, 193] tensor
        else:
            # distance: segmentation [b, 3, 200, 400]
            # distance = torch.zeros_like(scores_max).unsqueeze(1).expand(b, self.num_classes-1, scores_max.shape[1], scores_max.shape[2]) # zeros [b, 3, 200, 400]
            # dt_embedding = sample_dt(vertices_cell, F.sigmoid(distance), self.cell_size) # list of [N, 193] tensor

            # distance: feature [b, 256, 100, 200]
            distance_down = F.interpolate(distance, scale_factor=0.25, mode='bilinear', align_corners=True) # [b, 256, 25, 50]
            dt_embedding = sample_feat(vertices_cell, distance_down) # list of [N, 256] tensor

        if self.max_vertices >= 0:
            vertices, scores, dt_embedding, masks = list(zip(*[
                top_k_vertices(v, s, d, self.max_vertices)
                for v, s, d in zip(vertices, scores, dt_embedding)
            ]))

        # Convert (h, w) to (x, y), normalized
        # v: [N, 2]
        vertices_norm = [normalize_vertices(torch.flip(v, [1]).float(), score_shape) for v in vertices] # list of [N, 2] tensor
        
        # Vertices in pixel coordinate
        vertices = torch.stack(vertices).flip([2]) # [b, N, 2] x, y

        # Positional embedding (x, y, c)
        pos_embedding = [torch.cat((self.pe_fn(v), s.unsqueeze(1)), 1) for v, s in zip(vertices_norm, scores)] # list of [N, pe_dim+1] tensor
        pos_embedding = torch.stack(pos_embedding) # [b, N, pe_dim+1]

        # dt_embedding = [torch.cat((d, v, s.unsqueeze(1)), dim=1) for d, v, s in zip(dt_embedding, vertices_norm, scores)] # temp
        dt_embedding = torch.stack(dt_embedding) # [b, N, 64]
        masks = torch.stack(masks).unsqueeze(-1) # [b, N, 1]

        # graph_embedding = self.venc(pos_embedding) + self.dtenc(dt_embedding) if self.distance_reg else self.venc(pos_embedding) # [b, 256, N]
        graph_embedding = self.venc(pos_embedding) + self.dtenc(dt_embedding) # for visual descriptor
        # graph_embedding = self.dtenc(dt_embedding) # temp
        # masks = masks.transpose(1, 2) # [b, 1, N]
        graph_embedding = self.gnn(graph_embedding, masks.transpose(1, 2)) # [b, 256, N], [b, L, 4, N, N]
        graph_cls = self.cls_head(graph_embedding) # [b, 3, N]
        if self.refine:
            offset = torch.tanh(self.offset_head(graph_embedding)) # [b, 2, N]
        graph_embedding = self.final_proj(graph_embedding) # [b, 256, N]

        # Adjacency matrix score as inner product of all nodes
        matches = torch.einsum('bdn,bdm->bnm', graph_embedding, graph_embedding)
        matches = matches / self.feature_dim**.5 # [b, N, N] [match.fill_diagonal_(0.0) for match in matches]
        
        # Don't care self matches
        b, m, n = matches.shape
        diag_mask = torch.eye(m).repeat(b, 1, 1).bool()
        matches[diag_mask] = -1e9

        # Don't care bin matches
        match_mask = torch.einsum('bnd,bmd->bnm', masks, masks) # [B, N, N]
        matches = matches.masked_fill(match_mask == 0, -1e9)
        
        # Matching layer
        if self.sinkhorn_iters > 0:
            matches = log_optimal_transport(matches, self.bin_score, self.sinkhorn_iters) # [b, N+1, N+1]
        else:
            bins0 = self.bin_score.expand(b, m, 1) # [b, N, 1]
            bins1 = self.bin_score.expand(b, 1, n) # [b, 1, N]
            alpha = self.bin_score.expand(b, 1, 1) # [b, 1, 1]
            matches = torch.cat( # [b, N+1, N+1]
            [
                torch.cat([matches, bins0], -1), # [b, N, N+1]
                torch.cat([bins1, alpha], -1)   # [b, 1, N+1]
            ], 1)
            matches = F.log_softmax(matches, -1) # [b, N+1, N+1]
        # matches.exp() should be probability

        # Refinement offset in pixel coordinate
        if self.refine:
            _, h, w = score_shape
            offset = offset.permute(0, 2, 1)*offset.new_tensor([self.cell_size, self.cell_size]) # [b, N, 2] [-cell_size ~ cell_size]
            vertices = torch.clamp(vertices + offset, max=offset.new_tensor([w-1, h-1]), min=offset.new_tensor([0, 0]))

        # graph_cls = self.gcn(graph_embedding.transpose(1, 2), matches[:, :-1, :-1].exp()) # [b, N, num_classes]

        # return matches [b, N, N], vertices (pix coord) [b, N, 3], masks [b, N, 1]

        return F.log_softmax(graph_cls, 1), distance, vertex, instance, direction, (matches), vertices, masks
