from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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
    def __init__(self, xbound: list, ybound: list, match_threshold=0.2, reduction='mean') -> None:
        super(GraphLoss, self).__init__()
        
        # patch_size: [30.0, 60.0] list
        self.dx, self.bx, _ = gen_dx_bx(xbound, ybound)
        self.match_threshold = match_threshold
        self.reduction = reduction

        self.ce_fn = torch.nn.CrossEntropyLoss()
        self.nll_fn = torch.nn.NLLLoss()

    def forward(self, matches: torch.Tensor, positions: torch.Tensor, semantics: torch.Tensor, masks: torch.Tensor, vectors_gt: list):
        # matches: [b, N+1, N+1]
        # positions: [b, N, 2], x y
        # semantics: [b, 3, N] log_softmax dim=1
        # masks: [b, N, 1]
        # vectors_gt: [b] list of [instance] list of dict
        # matches = matches.exp()

        # iterate in batch
        cdist_list = []
        mloss_list = []
        semloss_list = []
        matches_gt = []
        semantics_gt = []
        for match, position, semantic, mask, vector_gt in zip(matches, positions, semantics, masks, vectors_gt):
            # match: [N, N+1]
            # position: [N, 2]
            # semantic: [3, N]
            # mask: [N, 1] M ones
            # vector_gt: [instance] list of dict
            mask = mask.squeeze(-1) # [N,]
            position_valid = position * torch.tensor(self.dx).cuda() + torch.tensor(self.bx).cuda() # de-normalize, [N, 2]
            position_valid = position_valid[mask == 1] # [M, 2] x, y
            pts_list = []
            pts_ins_list = []
            pts_type_list = []
            for ins, vector in enumerate(vector_gt): # dict
                pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
                pts = pts[:pts_num] # [p, 2] array
                [pts_list.append(pt) for pt in pts]
                [pts_ins_list.append(ins) for _ in pts] # instance ID for all vectors
                [pts_type_list.append(line_type) for _ in pts] # semantic for all vectors 0, 1, 2
            
            position_gt = torch.tensor(np.array(pts_list)).float().cuda() # [P, 2] shaped tensor
            match_gt = torch.zeros_like(match) # [N+1, N+1]
            semantic_gt = torch.zeros_like(semantic) # [3, N]

            if len(position_gt) > 0 and len(position_valid) > 0:            
                # compute chamfer distance # [N, P] shaped tensor
                cdist = torch.cdist(position_valid, position_gt) # [M, P]
                # nearest ground truth vectors
                nearest = cdist.argmin(-1) # [M,] shaped tensor, index of nearest position_gt
                cdist_mean = torch.mean(cdist[torch.arange(len(nearest)), nearest]) # mean of [N,] shaped tensor
                if len(nearest) > 1: # at least two vertices
                    dist_map = torch.cdist(position_valid, position_valid)
                    for idx_pred, idx_gt in enumerate(nearest):
                        # if idx_pred < len(nearest) - 1:
                        # idx_gt_with_same_ins = torch.where(torch.tensor(pts_ins_list).long().cuda() == pts_ins_list[idx_gt])[0] # can have more than one
                        # idx_pred -> idx_gt
                        semantic_gt[pts_type_list[idx_gt], idx_pred] = 1.0
                        
                        # find connection that shares identical nearest gt
                        # idx_pred_same = torch.where(nearest == idx_gt)[0]
                        # idx_pred_same = idx_pred_same[idx_pred_same != idx_pred]
                        # if len(idx_pred_same):
                        #     occupied, _ = match_gt[idx_pred_same].max(-1)
                        #     idx_pred_same = idx_pred_same[occupied < 1.0]
                        #     if len(idx_pred_same):
                        #         idx_pred_same = idx_pred_same[cdist[idx_pred_same, idx_gt].argmin()]
                        #         match_gt[idx_pred, idx_pred_same] = 1.0
                        #         continue
                        # find connection within same nearest gt instance iteratively
                        while True:
                            idx_gt_next = idx_gt + 1 if idx_gt < nearest.max() else -1
                            idx_pred_next = torch.where(nearest == idx_gt_next)[0]
                            idx_pred_next = idx_pred_next[cdist[idx_pred_next, idx_gt_next].argmin()] if len(idx_pred_next) else None # get one that has min distance
                            # match_gt[idx_pred, idx_pred_next] = 1.0 if idx_pred_next is not None and pts_ins_list[idx_gt] == pts_ins_list[idx_gt_next] else 0.0
                            
                            if idx_pred_next is not None and pts_ins_list[idx_gt] == pts_ins_list[idx_gt_next]:
                                match_gt[idx_pred, idx_pred_next] = 1.0
                                break
                            elif pts_ins_list[idx_gt] != pts_ins_list[idx_gt_next]:
                                break
                            else:
                                # match_gt[idx_pred, idx_pred_next] = 0.0
                                idx_gt = idx_gt_next


                        # # i: index of predicted vector, idx: index of gt vector nearest to the i-th predicted vector
                        # idx_prev = idx_gt - 1 if idx_gt > 0 else -1
                        # idx_next = idx_gt + 1 if idx_gt < nearest.max() else -1
                        # i_prev = torch.where(nearest == idx_prev)[0] # can have more than one
                        # i_next = torch.where(nearest == idx_next)[0] # can have more than one
                        # i_prev = i_prev[cdist[i_prev, idx_prev].argmin()] if len(i_prev) else None # get one that has min distance
                        # i_next = i_next[cdist[i_next, idx_next].argmin()] if len(i_next) else None # get one that has min distance
                        # match_gt[idx_pred, i_prev] = 1.0 if i_prev is not None and pts_ins_list[idx_gt] == pts_ins_list[idx_prev] else 0.0
                        # match_gt[idx_pred, i_next] = 1.0 if i_prev is not None and pts_ins_list[idx_gt] == pts_ins_list[idx_next] else 0.0
                    

                    match_gt_sum_backward = match_gt[:, :-1].sum(0) # [N]
                    # leave only one match along row dimension with closest vector
                    multi_cols, = torch.where(match_gt_sum_backward > 1) # [num_cols]
                    for multi_col in multi_cols:
                        rows, = torch.where(match_gt[:, multi_col] > 0)
                        match_gt[rows, multi_col] = 0.0
                        _, min_row_idx = dist_map[rows, multi_col].min(0)
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

                    match_loss = (match_loss_forward + match_loss_backward) * 0.5
                    # match_loss = match_loss_forward

                    semantic_valid = semantic[:, mask == 1].unsqueeze(0) # [1, 3, M]
                    semantic_gt_valid = semantic_gt[:, mask == 1].unsqueeze(0) # [1, 3, M]
                    assert torch.min(semantic_gt_valid.sum(1)) == 1, f"minimum value of semantic gt sum expected 1, but got: {torch.min(semantic_gt_valid.sum(1))}"
                    assert torch.max(semantic_gt_valid.sum(1)) == 1, f"maximum value of semantic gt sum expected 1, but got: {torch.max(semantic_gt_valid.sum(1))}"
                    semantic_gt_valid = semantic_gt_valid.argmax(1) # [1, M]

                    semantic_loss = self.nll_fn(semantic_valid, semantic_gt_valid)
                else:
                    match_loss = torch.tensor(0.0).float().cuda()
                    semantic_loss = torch.tensor(0.0).float().cuda()
            else:
                cdist_mean = torch.tensor(0.0).float().cuda()
                match_loss = torch.tensor(0.0).float().cuda()
                semantic_loss = torch.tensor(0.0).float().cuda()
            
            cdist_list.append(cdist_mean)
            mloss_list.append(match_loss)
            semloss_list.append(semantic_loss)
            matches_gt.append(match_gt)
            semantics_gt.append(semantic_gt)
        
        cdist_batch = torch.stack(cdist_list) # [b,]
        mloss_batch = torch.stack(mloss_list) # [b,]
        semloss_batch = torch.stack(semloss_list) # [b,]
        matches_gt = torch.stack(matches_gt) # [b, N+1, N+1]
        semantics_gt = torch.stack(semantics_gt) # [b, 3, N]

        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            cdist_batch = torch.mean(cdist_batch)
            mloss_batch = torch.mean(mloss_batch)
            semloss_batch = torch.mean(semloss_batch)
        elif self.reduction == 'sum':
            cdist_batch = torch.sum(cdist_batch)
            mloss_batch = torch.sum(mloss_batch)
            semloss_batch = torch.sum(semloss_batch)
        else:
            raise NotImplementedError
        
        return cdist_batch, mloss_batch, semloss_batch, matches_gt, semantics_gt


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
