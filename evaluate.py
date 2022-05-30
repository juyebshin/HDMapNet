import argparse
import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from tensorboardX import SummaryWriter

from data.dataset import semantic_dataset, vectormap_dataset
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou
from model import get_model
from model.vectormapnet import simple_nms
from data.visualize import colorise
from data.image import denormalize_img

colors_plt = ['r', 'b', 'g']

def onehot_encoding(logits, dim=1):
    # logits: b, C, 200, 400
    max_idx = torch.argmax(logits, dim, keepdim=True) # b, 1, 200, 400
    one_hot = logits.new_full(logits.shape, 0) # zeros b, C, 200, 400
    one_hot.scatter_(dim, max_idx, 1) # b, C, 200, 400 one hot
    return one_hot

def visualize(writer: SummaryWriter, title, imgs: torch.Tensor, dt_mask: torch.Tensor, vt_mask: torch.Tensor, vectors_gt: list, dt: torch.Tensor, heatmap: torch.Tensor, matches: torch.Tensor, positions: torch.Tensor, masks: torch.Tensor, patch_size: list, step: int):
    # imgs: b, 6, 3, 128, 352
    # dt: b, 3, 200, 400 tensor
    # heatmap: b, 65, 25, 50 tensor
    imgs = imgs.detach().cpu().float()[0] # 6, 3, 128, 352
    # imgs = imgs.fliplr()
    imgs = torch.index_select(imgs, 0, torch.LongTensor([0, 1, 2, 5, 4, 3]))
    imgs_grid = torchvision.utils.make_grid(imgs, nrow=3) # 3, 262, 1064
    imgs_grid = np.array(denormalize_img(imgs_grid)) # 262, 1064, 3
    writer.add_image(f'{title}/images', imgs_grid, step, dataformats='HWC')


    if dt is not None:
        dt = dt.detach().cpu().float().numpy()
        dt_mask = dt_mask.detach().cpu().float().numpy()
        writer.add_image(f'{title}/distance_transform_gt', colorise(dt_mask[0], 'magma'), step, dataformats='NHWC')
        writer.add_image(f'{title}/distance_transform_pred', colorise(dt[0], 'magma'), step, dataformats='NHWC')
    
    if heatmap is not None:
        vertex = onehot_encoding(heatmap)
        heatmap = heatmap.detach().cpu().float().numpy()[0] # 65, 25, 50
        vertex = vertex.detach().cpu().float().numpy()[0] # 65, 25, 50, onehot
        vt_mask = vt_mask.detach().cpu().float().numpy()[0] # 65, 25, 50

        nodust_gt = vt_mask[:-1, :, :] # 64, 25, 50
        Hc, Wc = vt_mask.shape[1:] # 25, 50
        nodust_gt = nodust_gt.transpose(1, 2, 0) # 25, 50, 64
        heatmap_gt = np.reshape(nodust_gt, [Hc, Wc, 8, 8]) # 25, 50, 8, 8
        heatmap_gt = np.transpose(heatmap_gt, [0, 2, 1, 3]) # 25, 8, 50, 8
        heatmap_gt = np.reshape(heatmap_gt, [Hc*8, Wc*8]) # 200, 400

        nodust = heatmap[:-1, :, :] # 64, 25, 50
        nodust = nodust.transpose(1, 2, 0) # 25, 50, 64
        heatmap = np.reshape(nodust, [Hc, Wc, 8, 8]) # 25, 50, 8, 8
        heatmap = np.transpose(heatmap, [0, 2, 1, 3]) # 25, 8, 50, 8
        heatmap = np.reshape(heatmap, [Hc*8, Wc*8]) # 200, 400
        # heatmap_nms = simple_nms(torch.from_numpy(heatmap).unsqueeze(0), 4).squeeze(0) # 200, 400

        nodust = vertex[:-1, :, :] # 64, 25, 50
        nodust = nodust.transpose(1, 2, 0) # 25, 50, 64
        vertex = np.reshape(nodust, [Hc, Wc, 8, 8]) # 25, 50, 8, 8
        vertex = np.transpose(vertex, [0, 2, 1, 3]) # 25, 8, 50, 8
        vertex = np.reshape(vertex, [Hc*8, Wc*8]) # 200, 400

        writer.add_image(f'{title}/vertex_heatmap_gt', colorise(heatmap_gt, 'hot', 0.0, 1.0), step, dataformats='HWC')
        writer.add_image(f'{title}/vertex_heatmap_pred', colorise(heatmap, 'hot', 0.0, 1.0), step, dataformats='HWC')
        writer.add_image(f'{title}/vertex_onehot_pred', colorise(vertex, 'hot', 0.0, 1.0), step, dataformats='HWC')
        heatmap[heatmap < 0.015] = 0.0
        heatmap[heatmap > 0.0] = 1.0
        writer.add_image(f'{title}/vertex_heatmap_bin', colorise(heatmap, 'hot', 0.0, 1.0), step, dataformats='HWC')
        # heatmap_nms[heatmap_nms < 0.015] = 0.0
        # heatmap_nms[heatmap_nms > 0.0] = 1.0
        # writer.add_image(f'{title}/vertex_heatmap_nms', colorise(heatmap_nms, 'hot', 0.0, 1.0), step, dataformats='HWC')
    
    if matches is not None and positions is not None and masks is not None:
        # matches: [b, N, N]
        # positions: [b, N, 3], x y c
        # masks: [b, 300, 1]
        # vectors_gt: [b] list of [instance] list of dict
        # patch_size: [30.0, 60.0]
        matches = matches.detach().cpu().float().numpy()[0] # [N, N]
        positions = positions.detach().cpu().float().numpy()[0] # [N, 3]
        masks = masks.detach().cpu().int().numpy()[0].squeeze(-1) # [N]
        vectors_gt = vectors_gt[0]
        positions[..., :-1] = positions[..., :-1] * np.array([patch_size[1], patch_size[0]])
        positions_valid = positions[masks == 1] # [M, 3]

        matches = np.triu(matches, 1)[masks == 1] # [N, N] upper triangle matrix without diagonal
        matches = matches[:, masks == 1] # upper masked [M, M]
        matches_idx = matches.argmax(1) # [M, ]

        fig = plt.figure(figsize=(4, 2))
        plt.xlim(-30, 30)
        plt.ylim(-15, 15)
        plt.axis('off')

        for vector in vectors_gt:
            pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
            pts = pts[:pts_num]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])
        
        writer.add_figure(f'{title}/vector_gt', fig, step)
        plt.close()

        fig = plt.figure(figsize=(4, 2))
        plt.xlim(-30, 30)
        plt.ylim(-15, 15)
        plt.axis('off')

        for i, pos in enumerate(positions_valid): # [3,]
            plt.scatter(pos[0], pos[1], s=0.5, color=colorise(pos[2], 'jet', 0.0, 1.0))
            match = matches_idx[i]
            if matches[i, match] > 0.8:
                plt.plot([pos[0], positions_valid[match][0]], [pos[1], positions_valid[match][1]], '-', color=colorise(pos[2], 'jet', 0.0, 1.0))
        
        writer.add_figure(f'{title}/vector_pred', fig, step)
        plt.close()



def eval_iou(model, val_loader, writer=None, step=None, vis_interval=0):
    # st
    model.eval()
    counter = 0
    total_intersects = 0
    total_union = 0
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, distance_gt, vertex_gt, vectors_gt in tqdm.tqdm(val_loader):

            semantic, distance, vertex, embedding, direction, matches, positions, masks = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())

            heatmap = vertex.softmax(1).cuda() # b, 65, 25, 50
            vertex_gt = vertex_gt.cuda().float() # b, 65, 25, 50
            intersects, union = get_batch_iou(onehot_encoding(heatmap), vertex_gt)
            total_intersects += intersects
            total_union += union

            if writer is not None and vis_interval > 0:
                if counter % vis_interval == 0:                
                        distance = distance.relu().clamp(max=10.0).cuda() # b, 3, 200, 400
                        # distance_gt = distance_gt.cuda() # b, 3, 200, 400
                        heatmap_onehot = onehot_encoding(heatmap)
                        # vertex_gt = vertex_gt.cuda().float() # b, 65, 25, 50
                        visualize(writer, 'eval', imgs, distance_gt, vertex_gt, vectors_gt, distance, heatmap, matches, positions, masks, [30.0, 60.0], step)
            
            counter += 1

    return total_intersects / (total_union + 1e-7)


def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'dist_threshold': args.dist_threshold, # 10.0
        'cell_size': args.cell_size, # 8
    }

    train_loader, val_loader = vectormap_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class, args.distance_reg)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()
    print(eval_iou(model, val_loader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='/home/user/data/Dataset/nuscenes/v1.0-trainval/')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

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

    # distance transform config
    parser.add_argument("--distance_reg", action='store_true')
    parser.add_argument("--dist_threshold", type=float, default=10.0)

    # vertex location classification config
    parser.add_argument("--vertex_pred", action='store_true')
    parser.add_argument("--cell_size", type=int, default=8)

    args = parser.parse_args()
    main(args)
