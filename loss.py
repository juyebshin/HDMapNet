from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment

def gen_dx_bx(xbound, ybound):
    dx = [row[2] for row in [xbound, ybound]] # [0.15, 0.15]
    bx = [row[0] + row[2] / 2.0 for row in [xbound, ybound]] # [-29.925, -14.925]
    nx = [(row[1] - row[0]) / row[2] for row in [xbound, ybound]] # [400, 200]
    return dx, bx, nx

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        # CE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce == 'mean':
            return torch.mean(F_loss)
        elif self.reduce == 'sum':
            return torch.sum(F_loss)
        else:
            raise NotImplementedError


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt): # b, 4, 200, 400
        loss = self.loss_fn(ypred, ytgt)
        return loss

# temp
class CEWithSoftmaxLoss(torch.nn.Module):
    def __init__(self):
        super(CEWithSoftmaxLoss, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, ypred, ytgt): # b, 65, 25, 50
        # ypred: b, 65, 25, 50
        # ytgt: b, 65, 25, 50 values [0-64)
        loss = self.loss_fn(ypred, ytgt)
        return loss

class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()
        self.loss_fn = torch.nn.NLLLoss()

    def forward(self, ypred, ytgt):
        # ypred: b, 65, 25, 50, onehot
        # ytgt: b, 65, 25, 50
        ytgt = torch.argmax(ytgt, dim=1) # b, 25, 50 values [0-64)
        loss = self.loss_fn(ypred, ytgt)
        return loss



class MSEWithReluLoss(torch.nn.Module):
    def __init__(self, dist_threshold=10.0):
        super(MSEWithReluLoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.dist_threshold = dist_threshold
    
    def forward(self, ypred, ytgt): # b, 3, 200, 400
        loss = self.loss_fn(torch.clamp(F.relu(ypred), max=self.dist_threshold), ytgt)
        return loss

class GraphLoss(nn.Module):
    def __init__(self, xbound: list, ybound: list, num_classes:int=3, cdist_threshold: float=1.5, reduction='mean', cost_class:float=1.0, cost_dist:float=5.0) -> None:
        super(GraphLoss, self).__init__()
        
        # patch_size: [30.0, 60.0] list
        self.dx, self.bx, self.nx = gen_dx_bx(xbound, ybound)
        self.bound = (np.array(self.dx)/2 - np.array(self.bx)) # [30.0, 15.0]
        self.num_classes = num_classes
        self.cdist_threshold = np.linalg.norm(cdist_threshold / (2*self.bound)) # norlamize distance threshold in meter / 45.0
        self.reduction = reduction
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1
        self.register_buffer('empty_weight', empty_weight)

        self.cost_class = cost_class
        self.cost_dist = cost_dist

        self.nll_fn = torch.nn.NLLLoss()

    def forward(self, matches: torch.Tensor, positions: torch.Tensor, semantics: torch.Tensor, masks: torch.Tensor, vectors_gt: list):
        # matches: [b, N+1, N+1]
        # positions: [b, N, 2], x y
        # semantics: [b, 3, N] log_softmax dim=1
        # masks: [b, N, 1]
        # vectors_gt: [b] list of [instance] list of dict
        # matches = matches.exp()

        # iterate in batch
        closs_list = []
        mloss_list = []
        semloss_list = []
        matches_gt = []
        semantics_gt = []
        for match, position, semantic, mask, vector_gt in zip(matches, positions, semantics, masks, vectors_gt):
            # match: [N, N+1]
            # position: [N, 2] pixel coords
            # semantic: [4, N]
            # mask: [N, 1] M ones
            # vector_gt: [instance] list of dict
            mask = mask.squeeze(-1) # [N,]
            position_valid = position / (torch.tensor(self.nx, device=position.device)-1) # normalize 0~1, [N, 2]
            position_valid = position_valid[mask == 1] # [M, 2] x, y
            semantic_valid = semantic[:, mask == 1] # [4, M]
            
            pts_list = []
            pts_ins_list = []
            pts_ins_order = []
            pts_type_list = []
            for ins, vector in enumerate(vector_gt): # dict
                pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
                pts = pts[:pts_num] # [p, 2] array
                # normalize coordinates 0~1
                [(pts_list.append((pt + self.bound) / (2*self.bound)), pts_ins_order.append(i)) for i, pt in enumerate(pts)]
                # [pts_list.append(pt) for pt in pts]
                [pts_ins_list.append(ins) for _ in pts] # instance ID for all vectors
                [pts_type_list.append(line_type) for _ in pts] # semantic for all vectors 0, 1, 2
            
            position_gt = torch.tensor(np.array(pts_list), device=position.device).float() # [P, 2] shaped tensor
            match_gt = torch.zeros_like(match) # [N+1, N+1]
            semantic_gt = torch.full(semantic.shape[1:], self.num_classes, dtype=torch.int64, device=semantic.device) # [N] 3: no class

            if len(position_gt) > 0 and len(position_valid) > 0:
                match_mat = torch.zeros([position_gt.shape[0], position_gt.shape[0]], dtype=torch.float, device=match.device)
                for k in range(max(pts_ins_list)+1):
                    ins_indices = [idx for idx, x in enumerate(pts_ins_list) if x == k]
                    match_mat[ins_indices[:-1], ins_indices[1:]] = 1.0
                # compute chamfer distance # [N, P] shaped tensor
                cdist = torch.cdist(position_valid, position_gt) # [M, P]
                cost_class = -semantic_valid.permute(1, 0)[:, pts_type_list] # [M, P]
                cost = self.cost_dist*cdist + self.cost_class*cost_class
                # nearest ground truth vectors
                pred_indices, gt_indices = linear_sum_assignment(cost.cpu().detach()) # bipartite matching, M indices
                # L1 loss
                coord_loss = F.l1_loss(position_valid[pred_indices], position_gt[gt_indices])
                # nearest_ins = []
                # for p, g in zip(pred_indices, gt_indices):
                #     nearest_ins.append(pts_ins_list[g] if cdist[p, g] < self.cdist_threshold else -1)
                #     semantic_gt[p] = pts_type_list[g] if cdist[p, g] < self.cdist_threshold else self.num_classes
                # for i in range(max(nearest_ins) + 1): # for all instance IDs
                #     indices = [ni for ni, x in enumerate(nearest_ins) if x == i] # ni: vector index, x: nearest instance ID
                #     ins_order = [pts_ins_order[gt_indices[oi]] for oi in indices]
                #     indices_sorted = [idx for ord, idx in sorted(zip(ins_order, indices))]
                #     match_gt[indices_sorted[:-1], indices_sorted[1:]] = 1.0
                
                # distance threshold
                thres_idx = (cdist[pred_indices, gt_indices] < self.cdist_threshold).cpu()
                if len(thres_idx) > 1: # at least two vertices
                    pred_indices = pred_indices[thres_idx] # thres_idx.cpu()
                    gt_indices = gt_indices[thres_idx]
                    match_mat = match_mat[gt_indices, :]
                    match_mat = match_mat[:, gt_indices] # [M, M]
                    for i1, pi1 in enumerate(pred_indices):
                        for i2, pi2 in enumerate(pred_indices):
                            match_gt[pi1, pi2] = match_mat[i1, i2]

                    semantic_gt[pred_indices] = torch.tensor(pts_type_list, device=semantic_gt.device)[gt_indices]
                    
                    match_gt_sum_backward = match_gt[:, :-1].sum(0) # [N]
                    # leave only one match along row dimension with closest vector
                    multi_cols, = torch.where(match_gt_sum_backward > 1) # [num_cols]
                    for multi_col in multi_cols:
                        rows, = torch.where(match_gt[:, multi_col] > 0)
                        match_gt[rows, multi_col] = 0.0
                        _, min_row_idx = cdist[rows, multi_col].min(0)
                        match_gt[rows[min_row_idx], multi_col] = 1.0
                    
                    mask_bins = torch.cat([mask, mask.new_tensor(1).expand(1)], 0)
                    match_gt_sum_forward = match_gt[:-1].sum(1) # [N]
                    match_gt[:-1][match_gt_sum_forward == 0, -1] = 1.0
                    assert torch.min(match_gt[:-1].sum(1)) == 1, f"minimum value of row-wise sum expected 1, but got: {torch.min(match_gt[:-1].sum(1))}"
                    assert torch.max(match_gt[:-1].sum(1)) == 1, f"maximum value of row-wise sum expected 1, but got: {torch.max(match_gt[:-1].sum(1))}"

                    match_gt_sum_backward = match_gt[:, :-1].sum(0)
                    match_gt[:, :-1][-1, match_gt_sum_backward == 0] = 1.0
                    assert torch.min(match_gt[:, :-1].sum(0)) == 1, f"minimum value of col-wise sum expected 1, but got: {torch.min(match_gt[:, :-1].sum(0))}"
                    assert torch.max(match_gt[:, :-1].sum(0)) == 1, f"maximum value of col-wise sum expected 1, but got: {torch.max(match_gt[:, :-1].sum(0))}"

                    match_valid = match[mask_bins == 1][:, mask_bins == 1] # [M+1, M+1]
                    match_gt_valid = match_gt[mask_bins == 1][:, mask_bins == 1] # [M, M+1]
                    assert torch.min(match_gt_valid[:-1].sum(1)) == 1, f"minimum value of row-wise sum expected 1, but got: {torch.min(match_gt_valid[:-1].sum(1))}"
                    assert torch.max(match_gt_valid[:-1].sum(1)) == 1, f"maximum value of row-wise sum expected 1, but got: {torch.max(match_gt_valid[:-1].sum(1))}"
                    assert torch.min(match_gt_valid[:, :-1].sum(0)) == 1, f"minimum value of col-wise sum expected 1, but got: {torch.min(match_gt_valid[:, :-1].sum(0))}"
                    assert torch.max(match_gt_valid[:, :-1].sum(0)) == 1, f"maximum value of col-wise sum expected 1, but got: {torch.max(match_gt_valid[:, :-1].sum(0))}"

                    # add minibatch dimension and class first
                    match_valid = match_valid.unsqueeze(0).transpose(1, 2) # [1, M+1, M+1] class dim first
                    match_gt_valid = match_gt_valid.unsqueeze(0) # [1, M+1, M+1]

                    # backward col -> row
                    match_gt_valid_backward = match_gt_valid.argmax(1) # col -> row [1, M+1]
                    match_loss_backward = self.nll_fn(match_valid[..., :-1], match_gt_valid_backward[..., :-1])

                    # forward row -> col
                    # match_valid = match_valid.transpose(1, 2) # [1, M+1, M+1]
                    match_gt_valid_forward = match_gt_valid.argmax(2) # row -> col [1, M+1]
                    match_loss_forward = self.nll_fn(match_valid[..., :-1], match_gt_valid_forward[..., :-1])

                    match_loss = (match_loss_forward + match_loss_backward)
                    # match_loss = match_loss_forward

                    semantic_valid = semantic_valid.unsqueeze(0) # [1, 4, M]
                    semantic_gt_valid = semantic_gt[mask == 1].unsqueeze(0) # [1, M]

                    semantic_loss = F.nll_loss(semantic_valid, semantic_gt_valid, self.empty_weight)
                else:
                    match_loss = position_gt.new_tensor(0.0)
                    semantic_loss = position_gt.new_tensor(0.0)
            else:
                coord_loss = position_gt.new_tensor(0.0)
                match_loss = position_gt.new_tensor(0.0)
                semantic_loss = position_gt.new_tensor(0.0)
            
            closs_list.append(coord_loss)
            mloss_list.append(match_loss)
            semloss_list.append(semantic_loss)
            matches_gt.append(match_gt)
            semantics_gt.append(semantic_gt)
        
        closs_batch = torch.stack(closs_list) # [b,]
        mloss_batch = torch.stack(mloss_list) # [b,]
        semloss_batch = torch.stack(semloss_list) # [b,]
        matches_gt = torch.stack(matches_gt) # [b, N+1, N+1]
        semantics_gt = torch.stack(semantics_gt) # [b, 3, N]

        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            closs_batch = torch.mean(closs_batch)
            mloss_batch = torch.mean(mloss_batch)
            semloss_batch = torch.mean(semloss_batch)
        elif self.reduction == 'sum':
            closs_batch = torch.sum(closs_batch)
            mloss_batch = torch.sum(mloss_batch)
            semloss_batch = torch.sum(semloss_batch)
        else:
            raise NotImplementedError
        
        return closs_batch, mloss_batch, semloss_batch, matches_gt, semantics_gt


class DiscriminativeLoss(nn.Module):
    def __init__(self, embed_dim, delta_v, delta_d):
        super(DiscriminativeLoss, self).__init__()
        self.embed_dim = embed_dim
        self.delta_v = delta_v
        self.delta_d = delta_d

    def forward(self, embedding, seg_gt):
        if embedding is None:
            return 0, 0, 0
        bs = embedding.shape[0]

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(bs):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim, 1), dim=0) - self.delta_v) ** 2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * self.delta_d  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / bs
        dist_loss = dist_loss / bs
        reg_loss = reg_loss / bs
        return var_loss, dist_loss, reg_loss


def calc_loss():
    pass
