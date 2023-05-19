import torch
import numpy as np
from einops import rearrange

from loss import gen_dx_bx


def get_batch_iou(pred_map, gt_map):
    # pred_map: [b, 3, 25, 50]
    # gt_map: [b, 3, 25, 50]
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[1]):
            pred = pred_map[:, i]
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float()
            union = (pred | tgt).sum().float()
            intersects.append(intersect)
            unions.append(union)
    return torch.tensor(intersects), torch.tensor(unions)

def get_batch_cd(pred_positions: torch.Tensor, gt_vectors: list, masks: torch.Tensor, xbound: list, ybound: list, nonsense: float = -1.0, threshold: float = 10.0):
    # pred_positions: b c N 3
    # gt_vectors: [b] list of [instance] list of dict
    # masks: b c N 1
    
    # pred_positions = rearrange(pred_positions, 'b c ... -> (b c) ...') # (b c) N 3
    # masks = rearrange(masks, 'b c ... -> (b c) ...') # (b c) N 1
    pred_positions = pred_positions[..., :-1] # (b c) N 2

    dx, bx, nx = gen_dx_bx(xbound, ybound)

    cdist_p_list = []
    cdist_l_list = []
    with torch.no_grad():
        for pred_position, gt_vector, mask in zip(pred_positions, gt_vectors, masks):
            # pred_position: c N 2
            # gt_vector: [instance] list of dict
            # mask: c N 1
            
            for ci, (cpred_position, cmask) in enumerate(zip(pred_position, mask)):
                # cpred_position: N 2
                # cmask: N 1
                cmask = cmask.squeeze(-1) # N
                cpred_position = cpred_position * torch.tensor(dx).cuda() + torch.tensor(bx).cuda() # de-normalize, [N, 2]
                cpred_position = cpred_position[cmask == 1] # M 2; x, y
                pts_list = []
                for ins, vector in enumerate(gt_vector): # dict
                    pts, pts_num, type = vector['pts'], vector['pts_num'], vector['type']
                    if type == ci:
                        pts = pts[:pts_num] # [p, 2] array
                        [pts_list.append(pt) for pt in pts]
            
                gt_position = torch.tensor(np.array(pts_list)).float().cuda() # P 2

                if len(gt_position) > 0 and len(cpred_position) > 0:
                    # compute chamfer distance # M P shaped tensor
                    cdist = torch.cdist(cpred_position, gt_position) # M P; prediction to label
                    # nearest ground truth vectors
                    cdist_p_mean = torch.mean(cdist.min(dim=-1).values) # M; prodiction to label
                    cdist_l_mean = torch.mean(cdist.min(dim=0).values) # P; label to prediction
                else:
                    cdist_p_mean = torch.tensor(nonsense).float().cuda()
                    cdist_l_mean = torch.tensor(nonsense).float().cuda()
            
                cdist_p_list.append(cdist_p_mean)
                cdist_l_list.append(cdist_l_mean)
        
        batch_cdist_p = torch.stack(cdist_p_list) # (b c)
        batch_cdist_l = torch.stack(cdist_l_list) # (b c)
        mask_p = batch_cdist_p != nonsense
        mask_l = batch_cdist_l != nonsense

        sum_p, sum_l = mask_p.sum(dim=0), mask_l.sum(dim=0)
        batch_cdist_p_mean = ((batch_cdist_p*mask_p).sum(dim=0) / sum_p) if sum_p > 0 else threshold
        batch_cdist_l_mean = ((batch_cdist_l*mask_l).sum(dim=0) / sum_l) if sum_l > 0 else threshold
    return batch_cdist_p_mean, batch_cdist_l_mean

def get_batch_vector_iou(pred_vectors: torch.Tensor, matches: torch.Tensor, masks: torch.Tensor, gt_map: torch.Tensor, thickness: int = 5):
    return None
