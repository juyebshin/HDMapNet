import os
import numpy as np
import sys
import logging
from time import time
from tensorboardX import SummaryWriter
import argparse

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from loss import NLLLoss, SimpleLoss, DiscriminativeLoss, MSEWithReluLoss, CEWithSoftmaxLoss, FocalLoss, GraphLoss

from data.dataset import semantic_dataset, vectormap_dataset
from data.const import NUM_CLASSES
from data.utils import is_main_process, label_onehot_decoding
from evaluation.iou import get_batch_iou, get_batch_cd
from evaluation.angle_diff import calc_angle_diff
from model import get_model
from evaluate import onehot_encoding, eval_iou, visualize

import data.utils as utils
import data.logger


def write_log(writer, ious, cdist, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)
    writer.add_scalar(f'{title}/cdist', cdist, counter)

    # for i, iou in enumerate(ious):
    #     writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)


def train(args):
    utils.init_distributed_mode(args)

    if not os.path.exists(args.logdir) and utils.is_main_process():
        os.mkdir(args.logdir)
    # logging.basicConfig(filename=os.path.join(args.logdir, "results.log"),
    #                     filemode='w',
    #                     format='%(asctime)s: %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S',
    #                     level=logging.INFO)
    # logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    # logger = logging.getLogger()
    # logger.addHandler(logging.StreamHandler(sys.stdout))

    logger = data.logger.setup_logger('vectorized map learning', args.logdir, utils.is_main_process(), "results.log")

    data_conf = {
        'num_channels': NUM_CLASSES + 1, # 4
        'image_size': args.image_size,
        'backbone': args.backbone,
        'xbound': args.xbound, # [-30.0, 30.0, 0.15]
        'ybound': args.ybound, # [-15.0, 15.0, 0.15]
        'zbound': args.zbound, # [-10.0, 10.0, 20.0]
        'dbound': args.dbound, # [4.0, 45.0, 1.0]
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
        'match_threshold': args.match_threshold, # 0.1
    }
    patch_size = [data_conf['ybound'][1] - data_conf['ybound'][0], data_conf['xbound'][1] - data_conf['xbound'][0]] # (30.0, 60.0)

    device = torch.device(args.device)
    BatchNorm1d = torch.nn.SyncBatchNorm if args.distributed else torch.nn.BatchNorm1d
    BatchNorm2d = torch.nn.SyncBatchNorm if args.distributed else torch.nn.BatchNorm2d
    norm_layer_dict = {'1d': BatchNorm1d, '2d': BatchNorm2d}
    
    if args.distributed:
        num_gpus = args.world_size
        assert args.bsz % num_gpus == 0, f"Check batch_size (batch_size % num_gpus == 0)"
        # args.lr = args.lr / 8 * args.bsz
        args.bsz = args.bsz // num_gpus
    
    train_loader, val_loader = vectormap_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers, args.distributed)
    model = get_model(args.model, data_conf, norm_layer_dict, args.segmentation, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class, args.distance_reg, args.vertex_pred, args.refine)
    model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.finetune:
        model_without_ddp.load_state_dict(torch.load(args.modelf), strict=False)
        # for name, param in model.named_parameters():
        #     if 'bev_backbone' in name:
        #         param.requires_grad = False
        #         logger.info("====="
        #             f"freezing {name}..."
        #             "=====")
        #     else:
        #         param.requires_grad = True

    opt = torch.optim.Adam(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = StepLR(opt, 10, 0.1)
    writer = SummaryWriter(logdir=args.logdir)

    loss_fn = SimpleLoss(args.pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).cuda()
    direction_loss_fn = torch.nn.BCELoss(reduction='none')
    dt_loss_fn = MSEWithReluLoss().cuda()
    vt_loss_fn = CEWithSoftmaxLoss().cuda()
    graph_loss_fn = GraphLoss(args.xbound, args.ybound).cuda()

    model.train()
    counter = 0
    best_iou = 0.0
    best_cd = 10.0
    last_idx = len(train_loader) - 1
    for epoch in range(args.nepochs):
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans,
                     yaw_pitch_roll, semantic_gt, instance_gt, distance_gt, vertex_gt, vectors_gt) in enumerate(train_loader):
            # vectors_gt: list of dict {'pts': array, 'pts_num': int, 'type': int}, each element of list is one instance of vectors
            t0 = time()
            opt.zero_grad()

            semantic, distance, vertex, embedding, direction, matches, positions, masks, attentions = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                   post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                   lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())

            semantic_gt = semantic_gt.cuda().float()
            instance_gt = instance_gt.cuda()
            distance_gt = distance_gt.cuda()
            vertex_gt = vertex_gt.cuda().float()

            vt_loss = vt_loss_fn(vertex, vertex_gt)

            if args.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(embedding, instance_gt)
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if args.direction_pred:
                direction_gt = direction_gt.cuda()
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(direction, direction_gt, args.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0
            
            if args.distance_reg:
                dt_loss = dt_loss_fn(distance, distance_gt)
            else:
                dt_loss = loss_fn(distance, semantic_gt[:, 1:])
                # normalize 0~1?
            
            cdist_loss, match_loss, seg_loss, matches_gt, vector_semantics_gt = graph_loss_fn(matches, positions, semantic, masks, vectors_gt)
            if not args.refine:
                cdist_loss = 0.0
            
            # if args.vertex_pred:
            #     # vertex_gt: b, 65, h, w
            #     vt_loss = vt_loss_fn(vertex, vertex_gt)
            # else:
            #     vt_loss = 0

            final_loss = seg_loss * args.scale_seg + var_loss * args.scale_var + dist_loss * args.scale_dist + direction_loss * args.scale_direction + dt_loss * args.scale_dt + vt_loss * args.scale_vt + cdist_loss * args.scale_cdist + match_loss * args.scale_match
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            # counter += 1
            t1 = time()

            if counter % 10 == 0:
                heatmap = vertex.softmax(1)
                intersects, union = get_batch_iou(onehot_encoding(heatmap), vertex_gt)
                iou = intersects / (union + 1e-7)
                cdist_p, cdist_l = get_batch_cd(positions, vectors_gt, masks, [-30.0, 30.0, 0.15], [-15.0, 15.0, 0.15])
                total_cdist = float((cdist_p + cdist_l)*0.5)
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1-t0:>7.4f}    "
                            f"Loss: {final_loss.item():>7.4f}    "
                            # f"IOU: {np.array2string(iou[:-1].numpy(), precision=3, floatmode='fixed')}    "
                            f"DT loss: {(dt_loss.item() if args.distance_reg else dt_loss):>7.4f}    "
                            f"Vertex loss: {vt_loss.item():>7.4f}    "
                            f"Match loss: {match_loss.item():>7.4f}    "
                            f"Seg loss: {seg_loss.item():>7.4f}    "
                            f"CD: {cdist_loss.item() if args.refine else cdist_loss:.4f}")

                write_log(writer, iou, total_cdist, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/direction_loss', direction_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)
                writer.add_scalar('train/angle_diff', angle_diff, counter)
                writer.add_scalar('train/dt_loss', dt_loss, counter)
                writer.add_scalar('train/vt_loss', vt_loss, counter)
                writer.add_scalar('train/match_loss', match_loss, counter)
                writer.add_scalar('train/cdist_loss', cdist_loss, counter)
                for bi, mask in enumerate(masks):
                    writer.add_scalar(f'train/num_vector_{bi}', torch.count_nonzero(mask), counter)
            
            if args.vis_interval > 0:
                if counter % args.vis_interval == 0 and utils.is_main_process():
                    if args.distance_reg:
                        distance = distance.relu().clamp(max=args.dist_threshold)
                    else:
                        distance = distance.sigmoid()
                    heatmap = vertex.softmax(1)
                    matches = matches.exp()
                    visualize(writer, 'train', imgs, distance_gt if args.distance_reg else semantic_gt[:, 1:], vertex_gt, vectors_gt, matches_gt, vector_semantics_gt, distance, heatmap, matches, positions, semantic, masks, attentions, args.xbound, args.ybound, counter)
                
            counter += 1

        iou, cdist = eval_iou(model, val_loader, writer, epoch, args.vis_interval, utils.is_main_process())
        logger.info(f"EVAL[{epoch:>2d}]:    "
                    # f"IOU: {np.array2string(iou[:-1].numpy(), precision=3, floatmode='fixed')}    "
                    f"CD: {cdist:.4f}")

        write_log(writer, iou, cdist, 'eval', epoch)
        # do not save this to save memory
        # model_name = os.path.join(args.logdir, f"model{epoch}.pt")
        # torch.save(model.state_dict(), model_name)
        # logger.info(f"{model_name} saved")
        # mean_iou = float(torch.mean(iou[:-1])) # mean excluding dustbin

        # # save best checkpoint
        # if mean_iou > best_iou:
        #     best_iou = mean_iou
        #     model_name = os.path.join(args.logdir, "model_best.pt")
        #     torch.save(model.state_dict(), model_name)
        #     logger.info(f"{model_name} saved")
        if cdist < best_cd:
            best_cd = cdist
            model_name = os.path.join(args.logdir, "model_best.pt")
            if utils.is_main_process():
                torch.save(model.module.state_dict() if args.distributed else model.state_dict(), model_name)
            logger.info(f"{model_name} saved")
        
        model.train()

        sched.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs/binary_logit_debug')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='./nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='VectorMapNet_cam')
    parser.add_argument("--backbone", type=str, default='efficientnet-b0',
                        choices=['efficientnet-b0', 'efficientnet-b4', 'efficientnet-b7', 'resnet-18', 'resnet-50'])

    # training config
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=12) # batch-size
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--vis_interval", type=int, default=200)

    # distributed training config
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--sample_dist", type=float, default=1.5) # 1.5

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=0.01) # vector segmentation
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)
    parser.add_argument("--scale_dt", type=float, default=1.0)
    parser.add_argument("--scale_vt", type=float, default=1.0)
    parser.add_argument("--scale_cdist", type=float, default=0.1, # 1.0
                        help="Scale of Chamfer distance loss")
    parser.add_argument("--scale_match", type=float, default=0.005, # 1.0
                        help="Scale of matching loss")

    # distance transform config
    parser.add_argument("--distance_reg", action='store_true') # store_true
    parser.add_argument("--dist_threshold", type=float, default=10.0)

    # vertex location classification config, always true for VectorMapNet
    parser.add_argument("--vertex_pred", action='store_false')
    parser.add_argument("--cell_size", type=int, default=8)

    # positional encoding frequencies
    parser.add_argument("--pos_freq", type=int, default=10,
                        help="log2 of max freq for positional encoding (2D vertex location)")

    # semantic segmentation config
    parser.add_argument("--segmentation", action='store_true')

    # vector refinement config
    parser.add_argument("--refine", action='store_true')

    # VectorMapNet config
    parser.add_argument("--num_vectors", type=int, default=400) # 100 * 3 classes = 300 in total
    parser.add_argument("--vertex_threshold", type=float, default=0.015)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", nargs='?', type=str, default=['self']*7)
    parser.add_argument("--sinkhorn_iterations", type=int, default=100)
    parser.add_argument("--match_threshold", type=float, default=0.1)

    args = parser.parse_args()
    train(args)
