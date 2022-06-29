import numpy as np
import torch
import torch.nn as nn

from .cluster import LaneNetPostProcessor
from .connect import sort_points_by_dist, connect_by_direction


def onehot_encoding(logits, dim=0):
    max_idx = torch.argmax(logits, dim, keepdim=True) # [1, 200, 400]
    one_hot = logits.new_full(logits.shape, 0) # [4, 200, 400]
    one_hot.scatter_(dim, max_idx, 1) # [4, 200, 400]
    return one_hot


def onehot_encoding_spread(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    one_hot.scatter_(dim, torch.clamp(max_idx-1, min=0), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx-2, min=0), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx+1, max=logits.shape[dim]-1), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx+2, max=logits.shape[dim]-1), 1)

    return one_hot


def get_pred_top2_direction(direction, dim=1):
    direction = torch.softmax(direction, dim)
    idx1 = torch.argmax(direction, dim)
    idx1_onehot_spread = onehot_encoding_spread(direction, dim)
    idx1_onehot_spread = idx1_onehot_spread.bool()
    direction[idx1_onehot_spread] = 0
    idx2 = torch.argmax(direction, dim)
    direction = torch.stack([idx1, idx2], dim) - 1
    return direction


def vectorize(segmentation, embedding, direction, angle_class):
    segmentation = segmentation.softmax(0) # [4, 200, 400]
    embedding = embedding.cpu() # [16, 200, 400]
    direction = direction.permute(1, 2, 0).cpu() # [200, 400, 37]
    direction = get_pred_top2_direction(direction, dim=-1) # [200, 400, 2]

    max_pool_1 = nn.MaxPool2d((1, 5), padding=(0, 2), stride=1)
    avg_pool_1 = nn.AvgPool2d((9, 5), padding=(4, 2), stride=1)
    max_pool_2 = nn.MaxPool2d((5, 1), padding=(2, 0), stride=1)
    avg_pool_2 = nn.AvgPool2d((5, 9), padding=(2, 4), stride=1)
    post_processor = LaneNetPostProcessor(dbscan_eps=1.5, postprocess_min_samples=50)

    oh_pred = onehot_encoding(segmentation).cpu().numpy() # [4, 200, 400]
    confidences = []
    line_types = []
    simplified_coords = []
    for i in range(1, oh_pred.shape[0]): # 1, 2, 3
        single_mask = oh_pred[i].astype('uint8') # [200, 400]
        single_embedding = embedding.permute(1, 2, 0) # [200, 400, 16]

        single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask, single_embedding) # [200, 400], [N, 2] 2: x, y
        if single_class_inst_mask is None:
            continue

        num_inst = len(single_class_inst_coords)

        prob = segmentation[i]
        prob[single_class_inst_mask == 0] = 0
        nms_mask_1 = ((max_pool_1(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        avg_mask_1 = avg_pool_1(prob.unsqueeze(0))[0].cpu().numpy()
        nms_mask_2 = ((max_pool_2(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        avg_mask_2 = avg_pool_2(prob.unsqueeze(0))[0].cpu().numpy()
        vertical_mask = avg_mask_1 > avg_mask_2
        horizontal_mask = ~vertical_mask
        nms_mask = (vertical_mask & nms_mask_1) | (horizontal_mask & nms_mask_2)

        for j in range(1, num_inst + 1):
            full_idx = np.where((single_class_inst_mask == j)) # [J], [J] row, col
            full_lane_coord = np.vstack((full_idx[1], full_idx[0])).transpose() # [J, 2]
            confidence = prob[single_class_inst_mask == j].mean().item()

            idx = np.where(nms_mask & (single_class_inst_mask == j)) # [K], [K] row, col
            if len(idx[0]) == 0:
                continue
            lane_coordinate = np.vstack((idx[1], idx[0])).transpose() # [K, 2]

            range_0 = np.max(full_lane_coord[:, 0]) - np.min(full_lane_coord[:, 0])
            range_1 = np.max(full_lane_coord[:, 1]) - np.min(full_lane_coord[:, 1])
            if range_0 > range_1:
                lane_coordinate = sorted(lane_coordinate, key=lambda x: x[0])
            else:
                lane_coordinate = sorted(lane_coordinate, key=lambda x: x[1])

            lane_coordinate = np.stack(lane_coordinate)
            lane_coordinate = sort_points_by_dist(lane_coordinate)
            lane_coordinate = lane_coordinate.astype('int32')
            lane_coordinate = connect_by_direction(lane_coordinate, direction, step=7, per_deg=360 / angle_class)

            simplified_coords.append(lane_coordinate)
            confidences.append(confidence)
            line_types.append(i-1)

    return simplified_coords, confidences, line_types

def vectorize_graph(positions: torch.Tensor, match: torch.Tensor, segmentation: torch.Tensor, mask: torch.Tensor, match_threshold=0.1):
    """ Vectorize from graph representations
    @ positions: [N, 2]
    @ match: [N+1, N+1]
    @ segmentation: [3, N] 
    @ mask: [N, 1] 
    @ patch_size: (30.0, 60.0)
    """
    assert match.shape[0] == match.shape[1], f"match.shape[0]: {match.shape[0]} != match.shape[1]: {match.shape[1]}"
    assert positions.shape[0] == segmentation.shape[1] == mask.shape[0], f"Following shapes mismatch: positions.shape[0]({positions.shape[0]}), segmentation.shape[1]({segmentation.shape[1]}), mask.shape[0]({mask.shape[0]}"

    mask = mask.squeeze(-1).cpu() # [N]
    mask_bin = torch.cat([mask, mask.new_tensor(1).view(1)], 0) # [N+1]
    match = match.exp().cpu()[mask_bin == 1][:, mask_bin == 1] # [M+1, M+1]
    positions = positions.cpu().numpy()[mask == 1] # [M, 3]
    # adj_mat = torch.zeros_like(match[:-1]) # [M, M+1]
    adj_mat = match[:-1, :-1] > 0.1 # [M, M] for > threshold
    # adj_mat = match[:-1] > 0.1 # [M, M+1] for argmax
    mscores, mindices = torch.topk(match[:-1], 2, -1) # [M, 2]? for top-2
    segmentation = segmentation.exp() # [3, N]
    seg_onehot = onehot_encoding(segmentation).cpu()[:, mask == 1].numpy() # [3, M] 0, 1, 2
    segmentation = segmentation.cpu().numpy()[:, mask == 1] # [3, M]

    confidences = []
    line_types = []
    simplified_coords = []
    for i in range(seg_onehot.shape[0]): # 0, 1, 2
        single_mask = np.expand_dims(seg_onehot[i].astype('uint8'), 1) # [M, 1]
        single_match_mask = single_mask @ single_mask.T # [M, M]

        single_class_adj_list = torch.nonzero(adj_mat & single_match_mask).numpy() # [M', 2] single_class_adj_list[:, 0] -> single_class_adj_list[:, 1]
        if single_class_adj_list.shape[0] == 0:
            continue
        single_class_adj_score = match[single_class_adj_list[:, 0], single_class_adj_list[:, 1]].numpy() # [M'] confidence

        prob = segmentation[i] # [M,]
        # prob = prob[single_class_adj_list[:, 0]] # [M,]

        while True:
            if single_class_adj_list.shape[0] == 0:
                break

            cur, next = single_class_adj_list[0] # cur -> next
            cur_idx, _ = np.where(single_class_adj_list[:, :-1] == cur)
            single_inst_coords = positions[cur] # [1, 2] np array
            single_inst_confidence = prob[cur] # [1] np array
            next_taken = [cur]
            cur_taken = next_taken.copy()
            del_idx = cur_idx

            while len(cur_idx):
                for ci in cur_idx:
                    cur, next = single_class_adj_list[ci] # cur -> next
                    cur = next
                    if cur not in cur_taken:
                        single_inst_coords = np.vstack((single_inst_coords, positions[cur])) # [K, 2]
                        single_inst_confidence = np.vstack((single_inst_confidence, prob[cur])) # [K, 1]
                        cur_taken.append(cur)
                next_taken.append(next)
                single_class_adj_list = np.delete(single_class_adj_list, del_idx, 0)
                cur_idx, _ = np.where(single_class_adj_list[:, :-1] == cur)
                next = single_class_adj_list[cur_idx, -1]
                del_idx = cur_idx
                next_taken_idx = []
                for j, n in enumerate(next):
                    if n in next_taken:
                        next_taken_idx.append(j)
                cur_idx = np.delete(cur_idx, next_taken_idx)
            
            # range_0 = np.max(single_inst_coords[:, 0]) - np.min(single_inst_coords[:, 0])
            # range_1 = np.max(single_inst_coords[:, 1]) - np.min(single_inst_coords[:, 1])
            # if range_0 > range_1:
            #     single_inst_coords = sorted(single_inst_coords, key=lambda x: x[0])
            # else:
            #     single_inst_coords = sorted(single_inst_coords, key=lambda x: x[1])
            
            # single_inst_coords = np.stack(single_inst_coords)
            # single_inst_coords = sort_points_by_dist(single_inst_coords)
            # single_inst_coords = single_inst_coords.astype('int32')
            
            simplified_coords.append(single_inst_coords)
            confidences.append(single_inst_confidence.mean())
            line_types.append(i)

        # cur, next = single_class_adj_list[0]
        # cur_idx, _ = np.where(single_class_adj_list[:, :-1] == cur)
        # if len(cur_idx) > 1:
        #     # max_idx_idx = match[single_class_adj_list[cur_idx, :-1].squeeze(-1), single_class_adj_list[cur_idx, -1]].argmax()
        #     max_idx_idx = single_class_adj_score[cur_idx].argmax()
        #     cur, next = single_class_adj_list[cur_idx[max_idx_idx]]
        # single_inst_coords = positions[cur] # [1, 2]
        # single_inst_confidence = prob[cur] # [1]
        # taken = [cur]
        # single_class_adj_list = np.delete(single_class_adj_list, cur_idx, 0)
        # single_class_adj_score = np.delete(single_class_adj_score, cur_idx, 0)
        # # prob = np.delete(prob, cur_idx, 0)
        
        # while True: # for all pairs, for instances?
        #     if single_class_adj_list.shape[0] == 0:
        #         break
            
        #     while True:
        #         cur = next
        #         cur_idx, _ = np.where(single_class_adj_list[:, :-1] == cur)
        #         del_idx = cur_idx
        #         next = single_class_adj_list[cur_idx, -1]
        #         for j, n in enumerate(next):
        #             if j < next.shape[0]:
        #                 next = np.delete(next, j) if n in taken else next
        #                 cur_idx = np.delete(cur_idx, j) if n in taken else cur_idx
        #         if len(next) < 1: # end of instance
        #             if single_class_adj_list.shape[0] > 0:
        #                 cur, next = single_class_adj_list[0]
        #                 cur_idx, _ = np.where(single_class_adj_list[:, :-1] == cur)
        #                 if len(cur_idx) > 1:
        #                     max_idx_idx = single_class_adj_score[cur_idx].argmax()
        #                     cur, next = single_class_adj_list[cur_idx[max_idx_idx]]
        #                 # sort by distance
        #                 # single_inst_coords = sort_points_by_dist(single_inst_coords)
        #                 if single_inst_coords.ndim == 1:
        #                     single_inst_coords = np.expand_dims(single_inst_coords, 0) # [1, 2]
        #                 simplified_coords.append(single_inst_coords)
        #                 confidences.append(single_inst_confidence.mean())
        #                 line_types.append(i)
        #                 single_inst_coords = positions[cur] # [1, 2]
        #                 single_inst_confidence = prob[cur] # [1]
        #                 taken.append(cur)
        #                 single_class_adj_list = np.delete(single_class_adj_list, cur_idx, 0)
        #                 single_class_adj_score = np.delete(single_class_adj_score, cur_idx, 0)
        #                 # prob = np.delete(prob, cur_idx, 0)
        #             break
        #         elif len(next) > 1:
        #             # max_idx_idx = match[single_class_adj_list[cur_idx, :-1].squeeze(-1), single_class_adj_list[cur_idx, -1]].argmax()
        #             max_idx_idx = single_class_adj_score[cur_idx].argmax()
        #             cur, next = single_class_adj_list[cur_idx[max_idx_idx]]
        #         if isinstance(next, np.ndarray):
        #             next = next[0]
        #         single_inst_coords = np.vstack((single_inst_coords, positions[cur])) # [K, 2]
        #         single_inst_confidence = np.vstack((single_inst_confidence, prob[cur])) # [K, 1]
        #         taken.append(cur)
        #         single_class_adj_list = np.delete(single_class_adj_list, del_idx, 0)
        #         single_class_adj_score = np.delete(single_class_adj_score, del_idx, 0)
                
                
                
                # prob = np.delete(prob, del_idx, 0)
            # simplified_coords.append(single_inst_coords)
    return simplified_coords, confidences, line_types
