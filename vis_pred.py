from builtins import print
import enum
import os
import argparse
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import tqdm
import torch

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def vis_segmentation(model, val_loader, logdir, distance_reg=False, dist_threshold=None, vertex_pred=False, cell_size=None, conf_threshold=0.015):
    semantic_color = np.array([
        [0, 0, 0], # background
        [0, 128, 0], # line
        [255, 255, 0], # ped_crossing
        [255, 0, 0] # contour
        ])
    dist_cmap = get_cmap('magma')
    vertex_cmap = get_cmap('hot')
    model.eval()
    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, distance_gt, vertex_gt) in enumerate(val_loader):

            semantic, distance, vertex, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            semantic = semantic.softmax(1).cpu().numpy() # b, 4, 200, 400
            distance = distance.relu().clamp(max=dist_threshold).cpu().numpy()
            vertex = vertex.softmax(1).cpu().numpy() # b, 65, 25, 50
            # semantic[semantic < 0.1] = np.nan
            semantic[semantic < 0.1] = 0.0

            semantic_gt = semantic_gt.cpu().numpy().astype('uint8')
            distance_gt = distance_gt.cpu().numpy().astype('float32')
            vertex_gt = vertex_gt.cpu().numpy().astype('uint8') * 255

            distance = (distance - 0.0) / (dist_threshold - 0.0)

            vmin = np.min(distance_gt)
            vmax = np.max(distance_gt)
            distance_gt = (distance_gt - vmin) / (vmax - vmin)

            for si in range(semantic.shape[0]): # iterate over batch
                # semantic: b, 4, 200, 400
                semantic_pred_onehot = np.argmax(semantic[si], axis=0)
                semantic_pred_color = semantic_color[semantic_pred_onehot].astype('uint8') # 200, 400, 3
                impath = os.path.join(logdir, 'seg')
                if not os.path.exists(impath):
                    os.mkdir(impath)
                imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                print('saving', imname)
                Image.fromarray(semantic_pred_color).save(imname)

                # semantic_gt: b, 4, 200, 400 value=[0, 1]
                semantic_gt_onehot = np.argmax(semantic_gt[si], axis=0)
                semantic_gt_color = semantic_color[semantic_gt_onehot].astype('uint8') # 200, 400, 3
                impath = os.path.join(logdir, 'seg_gt')
                if not os.path.exists(impath):
                    os.mkdir(impath)
                imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                print('saving', imname)
                Image.fromarray(semantic_gt_color).save(imname)

                # distance: b, 3, 200, 400
                if distance_reg:
                    distance_pred = np.max(distance[si], axis=0) # 200, 400
                    distance_pred_color = dist_cmap(distance_pred)[..., :3] * 255 # 200, 400, 3
                    impath = os.path.join(logdir, 'dist')
                    if not os.path.exists(impath):
                        os.mkdir(impath)
                    imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                    print('saving', imname)
                    Image.fromarray(distance_pred_color.astype('uint8')).save(imname)

                    # for idx, distance_single in enumerate(distance[si]): # for each class 0, 1, 2
                    #     distance_pred_color = dist_cmap(distance_single)[..., :3] * 255 # 200, 400, 3
                    #     imname = os.path.join(impath, f'eval{batchi:06}_{si:03}_{idx:01}.png')
                    #     print('saving', imname)
                    #     Image.fromarray(distance_pred_color.astype('uint8')).save(imname)

                    # distance_gt: b, 3, 200, 400
                    distance_gt_color = np.max(distance_gt[si], axis=0) # 200, 400
                    distance_gt_color = dist_cmap(distance_gt_color)[..., :3] * 255 # 200, 400, 3
                    impath = os.path.join(logdir, 'dist_gt')
                    if not os.path.exists(impath):
                        os.mkdir(impath)
                    imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                    print('saving', imname)
                    Image.fromarray(distance_gt_color.astype('uint8')).save(imname)

                if vertex_pred:
                    # heatmap
                    heatmap = vertex[si, :-1, :, :] # 64, 25, 50
                    Hc, Wc = heatmap.shape[1:] # 25, 50
                    heatmap = heatmap.transpose(1, 2, 0) # 25, 50, 64
                    heatmap = np.reshape(heatmap, [Hc, Wc, cell_size, cell_size]) # 25, 50, 8, 8
                    heatmap = np.transpose(heatmap, [0, 2, 1, 3]) # 25, 8, 50, 8
                    heatmap = np.reshape(heatmap, [Hc*cell_size, Wc*cell_size]) # 200, 400
                    heatmap[heatmap < conf_threshold] = 0.0 # 200, 400
                    heatmap_color = vertex_cmap(heatmap)[..., :3] * 255 # 200, 400, 3
                    impath = os.path.join(logdir, 'heatmap')
                    if not os.path.exists(impath):
                        os.mkdir(impath)
                    imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                    print('saving', imname)
                    Image.fromarray(heatmap_color.astype('uint8')).save(imname)
                    # vertex
                    # vertex_max = np.argmax(vertex[si], axis=0, keepdims=True) # 65, 25, 50 -> 1, 25, 50
                    # one_hot = np.full(vertex[si].shape, 0) # zeros 65, 25, 50
                    # np.put_along_axis(one_hot, vertex_max, 1, axis=0) # 65, 25, 50
                    heatmap[heatmap > 0] = 1
                    vertex_color = vertex_cmap(heatmap)[..., :3] * 255 # 200, 400, 3
                    impath = os.path.join(logdir, 'vertex')
                    if not os.path.exists(impath):
                        os.mkdir(impath)
                    imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                    print('saving', imname)
                    Image.fromarray(vertex_color.astype('uint8')).save(imname)
                    
                    nodust_gt = vertex_gt[si, :-1, :, :] # 64, 25, 50
                    Hc, Wc = nodust_gt.shape[1:] # 25, 50
                    nodust_gt = nodust_gt.transpose(1, 2, 0) # 25, 50, 64
                    heatmap_gt = np.reshape(nodust_gt, [Hc, Wc, cell_size, cell_size]) # 25, 50, 8, 8
                    heatmap_gt = np.transpose(heatmap_gt, [0, 2, 1, 3]) # 25, 8, 50, 8
                    heatmap_gt = np.reshape(heatmap_gt, [Hc*cell_size, Wc*cell_size]) # 200, 400
                    heatmap_gt_color = vertex_cmap(heatmap_gt)[..., :3] * 255 # 200, 400, 3
                    impath = os.path.join(logdir, 'vertex_gt')
                    if not os.path.exists(impath):
                        os.mkdir(impath)
                    imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                    print('saving', imname)
                    Image.fromarray(heatmap_gt_color.astype('uint8')).save(imname)

                # plt.figure(figsize=(4, 2))
                # plt.imshow(semantic[si][1], vmin=0, cmap='Blues', vmax=1)
                # plt.imshow(semantic[si][2], vmin=0, cmap='Reds', vmax=1)
                # plt.imshow(semantic[si][3], vmin=0, cmap='Greens', vmax=1)

                # # fig.axes.get_xaxis().set_visible(False)
                # # fig.axes.get_yaxis().set_visible(False)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')

                # impath = os.path.join(logdir, 'seg')
                # if not os.path.exists(impath):
                #     os.mkdir(impath)
                # imname = os.path.join(impath, f'train_seg{batchi:06}_{si:03}.jpg')
                # print('saving', imname)
                # plt.savefig(imname)
                # plt.close()

                # plt.figure(figsize=(4, 2))
                # plt.imshow(semantic_gt[si][1], vmin=0, cmap='Blues', vmax=1)
                # plt.imshow(semantic_gt[si][2], vmin=0, cmap='Reds', vmax=1)
                # plt.imshow(semantic_gt[si][3], vmin=0, cmap='Greens', vmax=1)

                # # fig.axes.get_xaxis().set_visible(False)
                # # fig.axes.get_yaxis().set_visible(False)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')

                # impath = os.path.join(logdir, 'seg_gt')
                # if not os.path.exists(impath):
                #     os.mkdir(impath)
                # imname = os.path.join(impath, f'train_seg{batchi:06}_{si:03}.jpg')
                # print('saving', imname)
                # plt.savefig(imname)
                # plt.close()

                # plt.figure(figsize=(4, 2))
                # plt.imshow(distance[si][0], vmin=0, cmap='magma', vmax=dist_threshold)
                # plt.imshow(distance[si][1], vmin=0, cmap='magma', vmax=dist_threshold)
                # plt.imshow(distance[si][2], vmin=0, cmap='magma', vmax=dist_threshold)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')

                # impath = os.path.join(logdir, 'dist')
                # if not os.path.exists(impath):
                #     os.mkdir(impath)
                # imname = os.path.join(impath, f'train_dist{batchi:06}_{si:03}.jpg')
                # print('saving', imname)
                # plt.savefig(imname)
                # plt.close()

                # plt.figure(figsize=(4, 2))
                # plt.imshow(distance_gt[si][0], vmin=0, cmap='magma', vmax=dist_threshold)
                # plt.imshow(distance_gt[si][1], vmin=0, cmap='magma', vmax=dist_threshold)
                # plt.imshow(distance_gt[si][2], vmin=0, cmap='magma', vmax=dist_threshold)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')

                # impath = os.path.join(logdir, 'dist_gt')
                # if not os.path.exists(impath):
                #     os.mkdir(impath)
                # imname = os.path.join(impath, f'train_dist{batchi:06}_{si:03}.jpg')
                # print('saving', imname)
                # plt.savefig(imname)
                # plt.close()


def vis_vector(model, val_loader, angle_class, logdir):
    model.eval()
    car_img = Image.open('icon/car.png')

    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt, distance_gt) in enumerate(val_loader):

            segmentation, distance, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                       post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                       lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())

            for si in range(segmentation.shape[0]):
                coords, _, _ = vectorize(segmentation[si], embedding[si], direction[si], angle_class)

                for coord in coords:
                    plt.plot(coord[:, 0], coord[:, 1], linewidth=5)

                plt.xlim((0, segmentation.shape[3]))
                plt.ylim((0, segmentation.shape[2]))
                plt.imshow(car_img, extent=[segmentation.shape[3]//2-15, segmentation.shape[3]//2+15, segmentation.shape[2]//2-12, segmentation.shape[2]//2+12])

                img_path = os.path.join(logdir, 'vector')
                if not os.path.exists(img_path):
                    os.mkdir(img_path)
                img_name = os.path.join(img_path, f'eval{batchi:06}_{si:03}.jpg')
                print('saving', img_name)
                plt.savefig(img_name)
                plt.close()


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

    train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()
    # vis_vector(model, val_loader, args.angle_class, args.logdir)
    vis_segmentation(model, val_loader, args.logdir, args.distance_reg, args.dist_threshold, args.vertex_pred, args.cell_size, args.conf_threshold)
    if args.instance_seg and args.direction_pred:
        vis_vector(model, val_loader, args.angle_class, args.logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='/home/user/data/Dataset/nuscenes/v1.0-trainval/')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='VectorMapNet_cam')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=12)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default='./runs/distance_vertex/model_best.pt')

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
    parser.add_argument("--conf_threshold", type=float, default=0.05)

    args = parser.parse_args()
    main(args)
