from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, patch_size: list, match_threshold=0.2, reduction='mean') -> None:
        super(GraphLoss, self).__init__()
        
        # patch_size: [30.0, 60.0] list
        self.patch_size = torch.tensor([patch_size[1], patch_size[0]]).cuda()
        self.match_threshold = match_threshold
        self.reduction = reduction

        self.mse_fn = torch.nn.MSELoss()

    def forward(self, matches: torch.Tensor, positions: torch.Tensor, masks: torch.Tensor, vectors_gt: list):
        # matches: [b, N, N]
        # positions: [b, N, 3], x y c
        # masks: [b, N, 1]
        # vectors_gt: [b] list of [instance] list of dict

        # iterate in batch
        cdist_list = []
        matches_gt = []
        mloss_list = []
        for match, position, mask, vector_gt in zip(matches, positions, masks, vectors_gt):
            # match: [N, N]
            # position: [N, 3]
            # mask: [N, 1] M ones
            # vector_gt: [instance] list of dict
            mask = mask.squeeze(-1) # [N,]
            position_valid = position[..., :-1] * self.patch_size # de-normalize, [N, 2]
            position_valid = position_valid[mask == 1] # [M, 2] x, y c
            pts_list = []
            pts_ins_list = []
            for ins, vector in enumerate(vector_gt): # dict
                pts, pts_num, type = vector['pts'], vector['pts_num'], vector['type']
                pts = pts[:pts_num] # [p, 2] array
                [pts_list.append(pt) for pt in pts]
                [pts_ins_list.append(ins) for _ in pts] # instance ID for all vectors
            
            position_gt = torch.tensor(pts_list).float().cuda() # [P, 2] shaped tensor
            match_gt = torch.zeros_like(match) # [N, N]

            if len(position_gt) > 0 and len(position_valid) > 0:            
                # compute chamfer distance # [N, P] shaped tensor
                cdist = torch.cdist(position_valid, position_gt) # [M, P]
                # nearest ground truth vectors
                nearest = cdist.argmin(-1) # [M,] shaped tensor, index of nearest position_gt
                cdist_mean = torch.mean(cdist[torch.arange(len(nearest)), nearest]) # mean of [N,] shaped tensor
                if len(nearest) > 1:
                    for idx_pred, idx_gt in enumerate(nearest):
                        # if idx_pred < len(nearest) - 1:
                        # idx_gt_with_same_ins = torch.where(torch.tensor(pts_ins_list).long().cuda() == pts_ins_list[idx_gt])[0] # can have more than one
                        idx_gt_next = idx_gt + 1 if idx_gt < nearest.max() else -1
                        idx_pred_next = torch.where(nearest == idx_gt_next)[0]
                        idx_pred_next = idx_pred_next[cdist[idx_pred_next, idx_gt_next].argmin()] if len(idx_pred_next) else None # get one that has min distance
                        match_gt[idx_pred, idx_pred_next] = 1.0 if idx_pred_next is not None and pts_ins_list[idx_gt] == pts_ins_list[idx_gt_next] else 0.0


                        # # i: index of predicted vector, idx: index of gt vector nearest to the i-th predicted vector
                        # idx_prev = idx_gt - 1 if idx_gt > 0 else -1
                        # idx_next = idx_gt + 1 if idx_gt < nearest.max() else -1
                        # i_prev = torch.where(nearest == idx_prev)[0] # can have more than one
                        # i_next = torch.where(nearest == idx_next)[0] # can have more than one
                        # i_prev = i_prev[cdist[i_prev, idx_prev].argmin()] if len(i_prev) else None # get one that has min distance
                        # i_next = i_next[cdist[i_next, idx_next].argmin()] if len(i_next) else None # get one that has min distance
                        # match_gt[idx_pred, i_prev] = 1.0 if i_prev is not None and pts_ins_list[idx_gt] == pts_ins_list[idx_prev] else 0.0
                        # match_gt[idx_pred, i_next] = 1.0 if i_prev is not None and pts_ins_list[idx_gt] == pts_ins_list[idx_next] else 0.0
                    
                    match = match[mask == 1][:, mask == 1] # [M, M]
                    # match_gt = torch.triu(match_gt, 1)
                    match_gt = torch.clamp(match_gt.T + match_gt, max=1.0) # Symmetry constraint
                    match_gt_valid = match_gt[mask == 1][:, mask == 1] # [M, M]
                    match_gt_valid[range(len(match_gt_valid)), range(len(match_gt_valid))] = match.diagonal() # ignore diagonal entities
                    match_loss = self.mse_fn(match, match_gt_valid)
                else:
                    match_loss = torch.tensor(0.0).float().cuda()
            else:
                cdist_mean = torch.tensor(0.0).float().cuda()
                match_loss = torch.tensor(0.0).float().cuda()
            
            cdist_list.append(cdist_mean)
            matches_gt.append(match_gt)
            mloss_list.append(match_loss)
        
        cdist_batch = torch.stack(cdist_list) # [b,]
        mloss_batch = torch.stack(mloss_list) # [b,]
        matches_gt = torch.stack(matches_gt) # [b,]

        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            cdist_batch = torch.mean(cdist_batch)
            mloss_batch = torch.mean(mloss_batch)
        elif self.reduction == 'sum':
            cdist_batch = torch.sum(cdist_batch)
            mloss_batch = torch.sum(mloss_batch)
        else:
            raise NotImplementedError
        
        return cdist_batch, mloss_batch, matches_gt


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
