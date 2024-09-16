import argparse
import tqdm
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision
import mmcv
from tensorboardX import SummaryWriter

from data.dataset import semantic_dataset, vectormap_dataset
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou, get_batch_cd
from model import get_model
from data.visualize import colorise, colors_plt
from data.image import denormalize_img
from loss import GraphLoss, gen_dx_bx
from postprocess.vectorize import vectorize_graph
from evaluate_json import get_val_info
from data.utils import all_gather
import data.logger
from os import path as osp
import cv2

def onehot_encoding(logits, dim=1):
    # logits: b, C, 200, 400
    max_idx = torch.argmax(logits, dim, keepdim=True) # b, 1, 200, 400
    one_hot = logits.new_full(logits.shape, 0) # zeros b, C, 200, 400
    one_hot.scatter_(dim, max_idx, 1) # b, C, 200, 400 one hot
    return one_hot

def visualize(writer: SummaryWriter, title, imgs: torch.Tensor, dt_mask: torch.Tensor, vt_mask: torch.Tensor, 
                vectors_gt: list, matches_gt: torch.Tensor, semantics_gt: torch.Tensor, depths_gt: torch.Tensor,
                dt: torch.Tensor, heatmap: torch.Tensor, matches: torch.Tensor, positions: torch.Tensor, semantics: torch.Tensor, depths: torch.Tensor,
                masks: torch.Tensor, step: int, args):
    if writer is None:
        return
    # imgs: b, 6, 3, 128, 352
    # dt: b, 3, 200, 400 tensor
    # heatmap: b, 65, 25, 50 tensor
    imgs = imgs.detach().cpu().float()[0] # 6, 3, 128, 352
    # imgs = imgs.fliplr()
    # imgs[3:] = torch.flip(imgs[3:], [3,])
    # imgs = torch.index_select(imgs, 0, torch.LongTensor([0, 1, 2, 5, 4, 3]))
    imgs_grid = torchvision.utils.make_grid(imgs, nrow=3) # 3, 262, 1064
    imgs_grid = np.array(denormalize_img(imgs_grid)) # 262, 1064, 3
    writer.add_image(f'{title}/images', imgs_grid, step, dataformats='HWC')

    dx, bx, nx = gen_dx_bx(args.xbound, args.ybound)

    # vectors_gt: [b] list of [instance] list of dict
    vectors_gt = vectors_gt[0]
    fig = plt.figure()
    plt.xlim(args.xbound[0], args.xbound[1])
    plt.ylim(args.ybound[0], args.ybound[1])
    plt.axis('off')

    for vector in vectors_gt:
        pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
        pts = pts[:pts_num]
        x = np.array([pt[0] for pt in pts])
        y = np.array([pt[1] for pt in pts])
        plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])
    
    writer.add_figure(f'{title}/vector_gt', fig, step)
    plt.close()


    if dt is not None:
        dt = dt.detach().cpu().float().numpy()
        dt_mask = dt_mask.detach().cpu().float().numpy()
        writer.add_image(f'{title}/distance_transform_gt', colorise(dt_mask[0], 'magma'), step, dataformats='NHWC')
        writer.add_image(f'{title}/distance_transform_pred', colorise(dt[0], 'magma'), step, dataformats='NHWC')
    
    if heatmap is not None:
        # vertex = onehot_encoding(heatmap)
        heatmap = heatmap.detach().cpu().float().numpy()[0] # 65, 25, 50
        # vertex = vertex.detach().cpu().float().numpy()[0] # 65, 25, 50, onehot
        vt_mask = vt_mask.detach().cpu().float().numpy()[0] # 65, 25, 50

        nodust_gt = vt_mask[:-1, :, :] # 64, 25, 50
        Hc, Wc = vt_mask.shape[1:] # 25, 50
        nodust_gt = nodust_gt.transpose(1, 2, 0) # 25, 50, 64
        heatmap_gt = np.reshape(nodust_gt, [Hc, Wc, args.cell_size, args.cell_size]) # 25, 50, 8, 8
        heatmap_gt = np.transpose(heatmap_gt, [0, 2, 1, 3]) # 25, 8, 50, 8
        heatmap_gt = np.reshape(heatmap_gt, [Hc*args.cell_size, Wc*args.cell_size]) # 200, 400

        nodust = heatmap[:-1, :, :] # 64, 25, 50
        nodust = nodust.transpose(1, 2, 0) # 25, 50, 64
        heatmap = np.reshape(nodust, [Hc, Wc, args.cell_size, args.cell_size]) # 25, 50, 8, 8
        heatmap = np.transpose(heatmap, [0, 2, 1, 3]) # 25, 8, 50, 8
        heatmap = np.reshape(heatmap, [Hc*args.cell_size, Wc*args.cell_size]) # 200, 400

        # nodust = vertex[:-1, :, :] # 64, 25, 50
        # nodust = nodust.transpose(1, 2, 0) # 25, 50, 64
        # vertex = np.reshape(nodust, [Hc, Wc, 8, 8]) # 25, 50, 8, 8
        # vertex = np.transpose(vertex, [0, 2, 1, 3]) # 25, 8, 50, 8
        # vertex = np.reshape(vertex, [Hc*8, Wc*8]) # 200, 400

        writer.add_image(f'{title}/vertex_heatmap_gt', colorise(heatmap_gt, 'hot', 0.0, 1.0), step, dataformats='HWC')
        # writer.add_image(f'{title}/vertex_heatmap_pred', colorise(heatmap, 'hot', 0.0, 1.0), step, dataformats='HWC')
        # writer.add_image(f'{title}/vertex_onehot_pred', colorise(vertex, 'hot', 0.0, 1.0), step, dataformats='HWC')
        heatmap[heatmap < args.vertex_threshold] = 0.0
        heatmap[heatmap > 0.0] = 1.0
        writer.add_image(f'{title}/vertex_heatmap_bin', colorise(heatmap, 'hot', 0.0, 1.0), step, dataformats='HWC')
    
    if matches is not None and positions is not None and masks is not None:
        # matches: [b, N+1, N+1]
        # positions: [b, N, 2], x y
        # masks: [b, N, 1]
        # attentions: [b, L, H, N, N] L: 7 layers, H: 4 heads
        # matches_gt: [b, N, N+1]
        # xbound: []
        # ybound: []
        matches = matches[0].detach().cpu().float().numpy() # [N+1, N+1]
        positions = positions[0].detach().cpu().float().numpy() # [N, 3]
        masks = masks[0].detach().cpu().int().numpy().squeeze(-1) # [N]
        # attentions = attentions[0].detach().cpu().float().numpy()[-1] # [H, N, N] attention of the last layer
        masks_bins = np.concatenate([masks, [1]], 0) # [N + 1]
        matches_gt = matches_gt.detach().cpu().float().numpy()[0] # [N+1, N+1]
        positions = positions * dx + bx # [30, 60]
        positions_valid = positions[masks == 1] # [M, 3]

        # Vector prediction
        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')
        # plt.scatter(positions_valid[:, 0], positions_valid[:, 1], s=0.5, c=positions_valid[:, 2], cmap='jet', vmin=0.0, vmax=1.0)

        matches = matches[masks_bins == 1][:, masks_bins == 1] # masked [M+1, M+1]
        matches_nodust = matches[:-1, :-1] # [M, M]
        rows, cols = np.where(matches_nodust > args.match_threshold)
        for row, col in zip(rows, cols):
            plt.plot([positions_valid[row, 0], positions_valid[col, 0]], [positions_valid[row, 1], positions_valid[col, 1]], 'o-', c=cm.jet(matches_nodust[row, col]), linewidth=0.5, markersize=1.0)
        # matches_idx = matches.argmax(1) if len(matches) > 0 else None # [M, ]
        # for i, pos in enumerate(positions_valid): # [3,]
        #     if matches_idx is not None:
        #         match = matches_idx[i]
        #         if matches[i, match] > 0.1 and match < len(matches)-1: # 0.8 too high?
        #             # plt.plot([pos[0], positions_valid[match][0]], [pos[1], positions_valid[match][1]], '-', color=colorise(matches[i, match], 'jet', 0.0, 1.0))
        #             plt.quiver(pos[0], pos[1], positions_valid[match][0] - pos[0], positions_valid[match][1] - pos[1], 
        #                        color=colorise(matches[i, match], 'jet', 0.0, 1.0), scale_units='xy', angles='xy', scale=1)
        
        writer.add_figure(f'{title}/vector_pred', fig, step)
        plt.close()

        # Attention
        # fig = plt.figure(figsize=(4, 2))
        # plt.xlim(args.xbound[0], args.xbound[1])
        # plt.ylim(args.ybound[0], args.ybound[1])
        # plt.axis('off')
        # # plt.scatter(positions_valid[:, 0], positions_valid[:, 1], s=0.5, c=positions_valid[:, 2], cmap='jet', vmin=0.0, vmax=1.0)
        # for attention in attentions: # [N, N] numpy
        #     attention = attention[masks == 1][:, masks == 1] # [M, M]
        #     if positions_valid.shape[0] > 0:
        #         values, indices = attention.max(-1), attention.argmax(-1) # [M]
        #         plt.quiver(positions_valid[:, 0], positions_valid[:, 1], positions_valid[indices, 0] - positions_valid[:, 0], positions_valid[indices, 1] - positions_valid[:, 1],
        #                    values, cmap='jet', scale_units='xy', angles='xy', scale=1)
        #     break # only for head 0
        # # plt.colorbar()
        # writer.add_figure(f'{title}/match_attention', fig, step)
        # plt.close()

        # Match prediction
        fig = plt.figure()
        plt.grid(False)
        plt.imshow(matches, cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0) # [M, M]
        plt.colorbar()
        writer.add_figure(f'{title}/match_pred', fig, step)
        plt.close()

        # Aligned GT matches
        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')

        # matches_gt = np.triu(matches_gt, 1)[masks == 1][:, masks_bins == 1] # [M, M] upper triangle matrix without diagonal
        matches_gt = matches_gt[masks_bins == 1][:, masks_bins == 1] # [M+1, M+1]
        # matches_gt = matches_gt[:, :-1] # [M, M]
        matches_idx = matches_gt.argmax(1) if len(matches_gt) > 0 else None # [M, ]
        
        plt.scatter(positions_valid[:, 0], positions_valid[:, 1], s=0.5, c='b', vmin=0.0, vmax=1.0)
        for i, pos in enumerate(positions_valid): # [3,]
            if matches_idx is not None:
                match = matches_idx[i]
                if matches_gt[i, match] == 1.0 and match < len(matches_gt)-1: # less then N
                    # plt.plot([pos[0], positions_valid[match][0]], [pos[1], positions_valid[match][1]], '-', color=colorise(matches_gt[i, match], 'jet', 0.0, 1.0))
                    plt.quiver(pos[0], pos[1], positions_valid[match][0] - pos[0], positions_valid[match][1] - pos[1], 
                               color=cm.jet(matches_gt[i, match]), scale_units='xy', angles='xy', scale=1)
        
        writer.add_figure(f'{title}/match_aligned', fig, step)
        plt.close()

        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')

        # Match gt
        fig = plt.figure()
        plt.grid(False)
        plt.imshow(matches_gt, cmap='hot', interpolation='nearest') # [M, M] !! row-wise sum is not always 1??
        plt.colorbar()
        writer.add_figure(f'{title}/match_gt', fig, step)
        plt.close()
    
    if semantics is not None and semantics_gt is not None:
        # semantics: [b, 3, N] 0: divider, 1: ped-crossing, 2: boundary
        # semantics_gt: [b, 3, N]
        semantic = semantics[0].exp().detach().cpu().float().numpy() # [3, N] if NLLLoss: .exp()
        semantic_gt = semantics_gt[0].detach().cpu().float().numpy().astype('uint8') # [3, N]
        
        semantic_onehot = semantic.argmax(0)[masks == 1] # [M]
        semantic_gt_onehot = semantic_gt.argmax(0)[masks == 1] # [M]

        # Semantic prediction
        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')
        plt.scatter(positions_valid[:, 0], positions_valid[:, 1], s=1.0, c=[colors_plt[c] for c in semantic_onehot])
        
        writer.add_figure(f'{title}/vector_semantic_pred', fig, step)
        plt.close()
        
        # Semantic gt
        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')
        plt.scatter(positions_valid[:, 0], positions_valid[:, 1], s=1.0, c=[colors_plt[c] for c in semantic_gt_onehot])
        
        writer.add_figure(f'{title}/vector_semantic_gt', fig, step)
        plt.close()
        
    if depths_gt is not None:
        B, _, _, _ = depths_gt.shape
        _, D, H, W = depths.shape
        depths = depths.view(B, -1, D, H, W)
        depth_gt = depths_gt[0].detach().cpu().float().numpy() # [N, H, W]
        depth = depths[0].detach().cpu().float().numpy() # [N, D, h, w]
        
        depth_onehot = depth.argmax(1)
        gt_depth_surround = np.zeros((depth_gt.shape[1]*2, depth_gt.shape[2]*3, 3), np.uint8)
        pred_depth_surround = np.zeros((depth_onehot.shape[1]*2, depth_onehot.shape[2]*3), np.float32)
        for idx, (img_, depth_gt_, depth_) in enumerate(zip(imgs, depth_gt, depth_onehot)):
            gt_depth_image = np.expand_dims(depth_gt_,2).repeat(3,2)
            im_color=cv2.applyColorMap(cv2.convertScaleAbs(gt_depth_image,alpha=15),cv2.COLORMAP_JET)
            image = np.array(denormalize_img(img_))
            image[gt_depth_image>0] = im_color[gt_depth_image>0]
            gt_depth_surround[image.shape[0]*(idx//3):image.shape[0]*(idx//3)+image.shape[0], 
                              image.shape[1]*(idx%3):image.shape[1]*(idx%3)+image.shape[1]] = image
            pred_depth_image = depth_.astype(np.float32) / (D-1)
            pred_depth_surround[pred_depth_image.shape[0]*(idx//3):pred_depth_image.shape[0]*(idx//3)+pred_depth_image.shape[0], 
                              pred_depth_image.shape[1]*(idx%3):pred_depth_image.shape[1]*(idx%3)+pred_depth_image.shape[1]] = pred_depth_image
        writer.add_image(f'{title}/depth_gt', gt_depth_surround, step, dataformats='HWC')
        writer.add_image(f'{title}/depth_pred', colorise(1.0-pred_depth_surround, 'jet', 0.0, 1.0), step, dataformats='HWC')



def eval_iou(model, val_loader, args, writer=None, step=None, vis_interval=0, is_master=False):
    # st
    graph_loss_fn = GraphLoss(args.xbound, args.ybound, num_classes=NUM_CLASSES).to(args.device)

    model.eval()
    counter = 0
    total_intersects = 0
    total_union = 0
    total_cdist_p = 0.0
    total_cdist_l = 0.0
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, distance_gt, vertex_gt, pv_semantic_gt, depth_gt, vectors_gt in tqdm.tqdm(val_loader):

            outputs = model(imgs.to(args.device), trans.to(args.device), rots.to(args.device), intrins.to(args.device),
                                                post_trans.to(args.device), post_rots.to(args.device), lidar_data.to(args.device),
                                                lidar_mask.to(args.device), car_trans.to(args.device), yaw_pitch_roll.to(args.device))
            
            if args.pv_seg:
                semantic, distance, vertex, matches, positions, masks, embedding, direction, depth, pv_seg = outputs
            else:
                semantic, distance, vertex, matches, positions, masks, embedding, direction, depth = outputs

            heatmap = vertex.softmax(1) # b, 65, 25, 50
            matches = matches.exp() # b, N+1, N+1
            vertex_gt = vertex_gt.to(args.device).float() # b, 65, 25, 50
            intersects, union = get_batch_iou(onehot_encoding(heatmap), vertex_gt)
            total_intersects += intersects
            total_union += union

            cdist_p, cdist_l = get_batch_cd(positions, vectors_gt, masks, args.xbound, args.ybound)
            total_cdist_p += cdist_p
            total_cdist_l += cdist_l

            _, _, seg_loss, matches_gt, vector_semantics_gt = graph_loss_fn(matches, positions, semantic, masks, vectors_gt)

            if writer is not None and vis_interval > 0:
                if counter % vis_interval == 0 and is_master:                
                        if args.distance_reg:
                            distance = distance.relu().clamp(max=args.dist_threshold).to(args.device) # b, 3, 200, 400
                        # distance_gt = distance_gt.to(args.device) # b, 3, 200, 400
                        heatmap_onehot = onehot_encoding(heatmap)
                        # vertex_gt = vertex_gt.to(args.device).float() # b, 65, 25, 50
                        visualize(writer, 'eval', imgs, distance_gt, vertex_gt, vectors_gt, matches_gt, vector_semantics_gt, depth_gt, distance, heatmap, matches, positions, semantic, depth, masks, step, args)
            
            counter += 1

    total_cdist_p, total_cdist_l = float(total_cdist_p/counter), float(total_cdist_l/counter)
    total_cdist = float((total_cdist_p + total_cdist_l)*0.5)
    print(f'CD_p: {total_cdist_p:.4f}, CD_l: {total_cdist_l:.4f}, CD: {total_cdist:.4f}')
    return total_intersects / (total_union + 1e-7), total_cdist

def eval_map(model, val_loader, args, writer=None, step=None, vis_interval=0, is_master=False):
    # st
    dx, bx, nx = gen_dx_bx(args.xbound, args.ybound)

    model.eval()
    counter = 0
    tokens_list = []
    results_list = []
    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, distance_gt, vertex_gt, pv_semantic_gt, depth_gt, vectors_gt) in enumerate(tqdm.tqdm(val_loader)):

            outputs = model(imgs.to(args.device), trans.to(args.device), rots.to(args.device), intrins.to(args.device),
                                                post_trans.to(args.device), post_rots.to(args.device), lidar_data.to(args.device),
                                                lidar_mask.to(args.device), car_trans.to(args.device), yaw_pitch_roll.to(args.device))
            
            if args.pv_seg:
                semantic, distance, vertex, matches, positions, masks, embedding, direction, depth, pv_seg = outputs
            else:
                semantic, distance, vertex, matches, positions, masks, embedding, direction, depth = outputs

            for si in range(imgs.shape[0]):
                coords, confidences, line_types = vectorize_graph(positions[si], matches[si], semantic[si], masks[si], args.match_threshold)
                result = {}
                vectors = []
                for coord, confidence, line_type in zip(coords, confidences, line_types):
                    vector = {'pts': coord * dx + bx, 'pts_num': len(coord), "type": line_type, "confidence_level": confidence}
                    vectors.append(vector)
                rec = val_loader.dataset.samples[batchi * val_loader.batch_size + si]
                tokens_list.append(rec['token'])
                results_list.append(vectors)

            if writer is not None and vis_interval > 0:
                if counter % vis_interval == 0 and is_master:  
                        # vertex_gt = vertex_gt.to(args.device).float() # b, 65, 25, 50
                        visualize(writer, 'eval', imgs, None, None, vectors_gt, None, None, None, None, None, None, None, None, results_list, step, args)
            
            counter += 1
        
    if is_master:
        submission = {
            "meta": {
                "use_camera": True,
                "use_lidar": False,
                "use_radar": False,
                "use_external": False,
                "vector": True,
            },
            "results": []
        }
        all_results_list = all_gather(results_list)
        all_tokens_list = all_gather(tokens_list)
        
        for tokens, results in zip(all_tokens_list, all_results_list):
            for token, result in zip(tokens, results):
                submission["results"].append({"sample_token": token, "vectors": result})
        
        args.result_path = osp.join(osp.join(*osp.split(args.modelf)[:-1]), 'vector_submission.json')
        mmcv.dump(submission, args.result_path)
        print(f"Results file saved to {args.result_path}")


def main(args):
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    logger = data.logger.setup_logger('vectorized map learning', args.logdir, True, "results.log")
    logger.info(args)
    
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'backbone': args.backbone,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'sample_dist': args.sample_dist, # 1.5
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'dist_threshold': args.dist_threshold, # 10.0
        'cell_size': args.cell_size, # 8
        'num_vectors': args.num_vectors, # 100
        'pos_freq': args.pos_freq, # 10
        'feature_dim': args.feature_dim, # 256
        'gnn_layers': args.gnn_layers, # ['self']*7
        'sinkhorn_iterations': args.sinkhorn_iterations, # 100
        'vertex_threshold': args.vertex_threshold, # 0.015
        'match_threshold': args.match_threshold, # 0.2
        'pv_seg': args.pv_seg,
        'pv_seg_classes': args.pv_seg_classes, # 1
        'feat_downsample': args.feat_downsample, # 16
        'depth_gt': args.depth_gt,
    }
    
    writer = SummaryWriter(logdir=args.logdir)

    BatchNorm1d = torch.nn.BatchNorm1d
    BatchNorm2d = torch.nn.BatchNorm2d
    norm_layer_dict = {'1d': BatchNorm1d, '2d': BatchNorm2d}
    
    train_loader, val_loader = vectormap_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model, data_conf, norm_layer_dict, args.segmentation, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class, args.distance_reg, args.vertex_pred, args.refine)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.to(args.device)
    print(eval_iou(model, val_loader, args, writer, 0, args.vis_interval, True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='./nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='VectorMapNet_cam')
    parser.add_argument("--backbone", type=str, default='efficientnet-b4',
                        choices=['efficientnet-b0', 'efficientnet-b4', 'efficientnet-b7', 'resnet-18', 'resnet-50'])

    # training config
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--vis_interval", type=int, default=200)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default='./runs/graph_debug_v2/model_best.pt')

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--sample_dist", type=float, default=1.5)

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)
    parser.add_argument("--scale_dt", type=float, default=1.0)
    parser.add_argument("--scale_cdist", type=float, default=1.0, 
                        help="Scale of Chamfer distance loss")
    parser.add_argument("--scale_match", type=float, default=1.0, 
                        help="Scale of matching loss")

    # distance transform config
    parser.add_argument("--distance_reg", action='store_true')
    parser.add_argument("--dist_threshold", type=float, default=10.0)

    # vertex location classification config
    parser.add_argument("--vertex_pred", action='store_false')
    parser.add_argument("--cell_size", type=int, default=8)
    
    # pv segmentation config
    parser.add_argument("--pv_seg", action='store_true')
    parser.add_argument("--pv_seg_classes", type=int, default=1)
    parser.add_argument("--feat_downsample", type=int, default=16)
    
    # depth map config
    parser.add_argument("--depth-gt", action='store_true')

    # positional encoding frequencies
    parser.add_argument("--pos_freq", type=int, default=10,
                        help="log2 of max freq for positional encoding (2D vertex location)")

    # semantic segmentation config
    parser.add_argument("--segmentation", action='store_true')

    # vector refinement config
    parser.add_argument("--refine", action='store_true')

    # VectorMapNet config
    parser.add_argument("--num_vectors", type=int, default=300) # 100 * 3 classes = 300 in total
    parser.add_argument("--vertex_threshold", type=float, default=0.01)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=7)
    parser.add_argument("--sinkhorn_iterations", type=int, default=100)
    parser.add_argument("--match_threshold", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
