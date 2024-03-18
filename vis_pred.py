from builtins import print
import enum
import os
import argparse
import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgb

import tqdm
import torch
import torchvision
from yaml import parse

from data.dataset import semantic_dataset, vectormap_dataset, VectorMapNetDataset
from data.const import CAMS, NUM_CLASSES
from data.utils import get_proj_mat, perspective
from model import get_model
from postprocess.vectorize import vectorize, vectorize_graph

from data.image import denormalize_img
from data.visualize import colorise, colors_plt
from export_pred_to_json import gen_dx_bx


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
    car_img = Image.open('icon/car.png')
    dist_cmap = get_cmap('magma')
    vertex_cmap = get_cmap('hot')
    xbound, ybound = [-30.0, 30.0, 0.15], [-15.0, 15.0, 0.15]
    dx, bx, nx = gen_dx_bx(xbound, ybound)
    model.eval()
    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, distance_gt, vertex_gt, vectors_gt) in enumerate(val_loader):

            semantic, distance, vertex, embedding, direction, matches, positions, masks = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            # semantic = semantic.softmax(1).cpu().numpy() # b, 4, 200, 400
            distance = distance.relu().clamp(max=dist_threshold).cpu().numpy()
            vertex = vertex.softmax(1).cpu().numpy() # b, 65, 25, 50
            matches_top2, indices_top2 = torch.topk(matches.exp(), 2, -1) # [b, N+1, 2]
            matches = matches.exp().cpu().float().numpy() # b, N+1, N+1 for sinkhorn
            masks = masks.detach().cpu().int().numpy().squeeze(-1) # b, 300
            # attentions = attentions.detach().cpu().float().numpy() # b, 7, 4, 300, 300
            # semantic[semantic < 0.1] = np.nan
            # semantic[semantic < 0.1] = 0.0

            semantic_gt = semantic_gt.cpu().numpy().astype('uint8')
            distance_gt = distance_gt.cpu().numpy().astype('float32')
            vertex_gt = vertex_gt.cpu().numpy().astype('uint8') * 255

            distance = (distance - 0.0) / (dist_threshold - 0.0)

            vmin = np.min(distance_gt)
            vmax = np.max(distance_gt)
            distance_gt = (distance_gt - vmin) / (vmax - vmin)

            for si in range(imgs.shape[0]): # iterate over batch
                idx = batchi*val_loader.batch_size + si
                rec = val_loader.dataset.samples[idx]
                scene_name = val_loader.dataset.nusc.get('scene', rec['scene_token'])['name']
                lidar_top_path = val_loader.dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])
                base_name = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0].split('_')[-1] # timestamp
                base_name = scene_name + '_' + base_name # {scene_name}_{timestamp}
                # semantic: b, 4, 200, 400
                # semantic_pred_onehot = np.argmax(semantic[si], axis=0)
                # semantic_pred_color = semantic_color[semantic_pred_onehot].astype('uint8') # 200, 400, 3
                # impath = os.path.join(logdir, 'seg')
                # if not os.path.exists(impath):
                #     os.mkdir(impath)
                # imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                # print('saving', imname)
                # Image.fromarray(semantic_pred_color).save(imname)

                # # semantic_gt: b, 4, 200, 400 value=[0, 1]
                # semantic_gt_onehot = np.argmax(semantic_gt[si], axis=0)
                # semantic_gt_color = semantic_color[semantic_gt_onehot].astype('uint8') # 200, 400, 3
                # impath = os.path.join(logdir, 'seg_gt')
                # if not os.path.exists(impath):
                #     os.mkdir(impath)
                # imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                # print('saving', imname)
                # Image.fromarray(semantic_gt_color).save(imname)

                # imgs: b, 6, 3, 128, 352
                # img = imgs[si].detach().cpu().float() # 6, 3, 128, 352
                # img[3:] = torch.flip(img[3:], [3,])
                # img_grid = torchvision.utils.make_grid(img, nrow=3) # 3, 262, 1064
                # img_grid = np.array(denormalize_img(img_grid)) # 262, 1064, 3
                # impath = os.path.join(logdir, 'images')
                # if not os.path.exists(impath):
                #     os.mkdir(impath)
                # imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                # print('saving', imname)
                # Image.fromarray(img_grid).save(imname)

                # distance: b, 3, 200, 400
                if distance_reg:
                    distance_pred = np.max(distance[si], axis=0) # 200, 400
                    distance_pred_color = dist_cmap(distance_pred)[..., :3] * 255 # 200, 400, 3
                    impath = os.path.join(logdir, 'dist')
                    if not os.path.exists(impath):
                        os.mkdir(impath)
                    imname = os.path.join(impath, f'{base_name}.png')
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
                    imname = os.path.join(impath, f'{base_name}.png')
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
                    imname = os.path.join(impath, f'{base_name}.png')
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
                    imname = os.path.join(impath, f'{base_name}.png')
                    print('saving', imname)
                    Image.fromarray(vertex_color.astype('uint8')).save(imname)
                    
                    # nodust_gt = vertex_gt[si, :-1, :, :] # 64, 25, 50
                    # Hc, Wc = nodust_gt.shape[1:] # 25, 50
                    # nodust_gt = nodust_gt.transpose(1, 2, 0) # 25, 50, 64
                    # heatmap_gt = np.reshape(nodust_gt, [Hc, Wc, cell_size, cell_size]) # 25, 50, 8, 8
                    # heatmap_gt = np.transpose(heatmap_gt, [0, 2, 1, 3]) # 25, 8, 50, 8
                    # heatmap_gt = np.reshape(heatmap_gt, [Hc*cell_size, Wc*cell_size]) # 200, 400
                    # heatmap_gt_color = vertex_cmap(heatmap_gt)[..., :3] * 255 # 200, 400, 3
                    # impath = os.path.join(logdir, 'vertex_gt')
                    # if not os.path.exists(impath):
                    #     os.mkdir(impath)
                    # imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                    # print('saving', imname)
                    # Image.fromarray(heatmap_gt_color.astype('uint8')).save(imname)
                    
                    # # ground truth vectors
                    # vector_gt = vectors_gt[si] # [instance] list of dict

                    # impath = os.path.join(logdir, 'vector_gt')
                    # if not os.path.exists(impath):
                    #     os.mkdir(impath)
                    # imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                    # print('saving', imname)

                    # fig = plt.figure(figsize=(4, 2))
                    # plt.xlim(-30, 30)
                    # plt.ylim(-15, 15)
                    # plt.axis('off')

                    # for vector in vector_gt:
                    #     pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
                    #     pts = pts[:pts_num]
                    #     x = np.array([pt[0] for pt in pts])
                    #     y = np.array([pt[1] for pt in pts])
                    #     plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])
                    # plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
                    # plt.savefig(imname, bbox_inches='tight', dpi=400)
                    # plt.close()

                    # vector prediction
                    mask = masks[si] # [400]
                    position_valid = positions[si].detach().cpu().float().numpy() # [N, 3]
                    position_valid = position_valid * dx + bx # [-30, -15, 30, 15]
                    position_valid = position_valid[mask == 1] # [M, 3]

                    # vector segmentation prediction
                    semantic_onehot = semantic[si].exp().detach().cpu().float().numpy() # [3, N]
                    semantic_onehot = semantic_onehot.argmax(0)[mask == 1] # [M]

                    fig = plt.figure(figsize=(4, 2))
                    plt.xlim(-30, 30)
                    plt.ylim(-15, 15)
                    plt.axis('off')
                    plt.scatter(position_valid[:, 0], position_valid[:, 1], s=1.0, c=[colors_plt[c] for c in semantic_onehot])
                    
                    impath = os.path.join(logdir, 'segmentation')
                    if not os.path.exists(impath):
                        os.mkdir(impath)
                    imname = os.path.join(impath, f'{base_name}.png')
                    print('saving', imname)
                    plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
                    plt.savefig(imname, bbox_inches='tight', pad_inches=0, dpi=400)
                    plt.close()

                    # # matches
                    # mask_bin = np.concatenate([mask, [1]], 0) # [N + 1]
                    # match = matches[si] # [N, N+1]
                    # match = match[mask_bin == 1][:, mask_bin == 1] # [M+1, M+1]
                    # match_idx = match[:-1].argmax(1) if len(match) > 0 else None # [M]
                    # match = match[:-1, :-1] # [M, M] no dust
                    # rows, cols = np.where(match > 0.1)
                    
                    # impath = os.path.join(logdir, 'vector_pred')
                    # if not os.path.exists(impath):
                    #     os.mkdir(impath)
                    # imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                    # print('saving', imname)

                    # fig = plt.figure(figsize=(4, 2))
                    # plt.xlim(-30, 30)
                    # plt.ylim(-15, 15)
                    # plt.axis('off')
                    # plt.grid(False)

                    # # plt.scatter(position_valid[:, 0], position_valid[:, 1], s=1.0, c=position_valid[:, 2], cmap='jet', vmin=0.0, vmax=1.0)
                    # for row, col in zip(rows, cols):
                    #     plt.plot([position_valid[row, 0], position_valid[col, 0]], [position_valid[row, 1], position_valid[col, 1]], 'o-', c=colorise(match[row, col], 'jet', 0.0, 1.0), linewidth=0.5, markersize=1.0)
                    # plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
                    # plt.savefig(imname, bbox_inches='tight', dpi=400)
                    # plt.close()

                    match_top2, index_top2 = matches_top2[si, :-1].cpu().numpy()[mask==1], indices_top2[si, :-1].cpu().numpy()[mask==1] # [M, 2]
                    
                    impath = os.path.join(logdir, 'vector_top2')
                    if not os.path.exists(impath):
                        os.mkdir(impath)
                    imname = os.path.join(impath, f'{base_name}.png')
                    print('saving', imname)

                    fig = plt.figure(figsize=(4, 2))
                    plt.xlim(-30, 30)
                    plt.ylim(-15, 15)
                    plt.axis('off')
                    plt.grid(False)

                    # plt.scatter(position_valid[:, 0], position_valid[:, 1], s=0.5, c=position_valid[:, 2], cmap='jet', vmin=0.0, vmax=1.0)
                    for idx, (score, next) in enumerate(zip(match_top2, index_top2)):
                        for s, n in zip(score, next):
                            if n < len(position_valid):
                                plt.plot([position_valid[idx, 0], position_valid[n, 0]], [position_valid[idx, 1], position_valid[n, 1]], '-', c=colorise(s, 'jet', 0.0, 1.0))
                    
                    plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
                    plt.savefig(imname, bbox_inches='tight', dpi=400)
                    plt.close()

                    # attention = attentions[si, -1] # [4, 300, 300]

                    # impath = os.path.join(logdir, 'match_attention')
                    # if not os.path.exists(impath):
                    #     os.mkdir(impath)
                    # imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                    # print('saving', imname)

                    # fig=plt.figure(figsize=(4, 2))
                    # plt.xlim(-30, 30)
                    # plt.ylim(-15, 15)
                    # plt.axis('off')
                    # plt.grid(False)
                    # plt.scatter(position_valid[:, 0], position_valid[:, 1], s=0.5, c=position_valid[:, 2], cmap='jet', vmin=0.0, vmax=1.0)
                    # for attention_head in attention: # [N, N] numpy
                    #     attention_head = attention_head[mask == 1][:, mask == 1] # [M, M]
                    #     if position_valid.shape[0] > 0:
                    #         values, indices = attention_head.max(-1), attention_head.argmax(-1) # [M]
                    #         plt.quiver(position_valid[:, 0], position_valid[:, 1], position_valid[indices, 0] - position_valid[:, 0], position_valid[indices, 1] - position_valid[:, 1],
                    #                    values, cmap='jet', scale_units='xy', angles='xy', scale=1)
                    #     break # only for head 0
                    # plt.colorbar()
                    # plt.savefig(imname, bbox_inches='tight', dpi=400)
                    # plt.close()


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

def vis_vectormapnet(model, val_loader, logdir, data_conf):
    model.eval()
    car_img = Image.open('icon/car.png')
    xbound, ybound = data_conf['xbound'], data_conf['ybound']
    dx, bx, nx = gen_dx_bx(xbound, ybound)
    img_size = data_conf['image_size'] # [128, 352]
    intrin_scale = torch.tensor([img_size[1] / 1600., img_size[0] / 900., 1.0]) * torch.eye(3)

    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, distance_gt, vertex_gt, vectors_gt) in enumerate(val_loader):
            
            semantic, distance, vertex, embedding, direction, matches, positions, masks = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            
            for si in range(imgs.shape[0]):
                coords, confidences, line_types = vectorize_graph(positions[si], matches[si], semantic[si], masks[si], data_conf['match_threshold'])
                idx = batchi*val_loader.batch_size + si
                rec = val_loader.dataset.samples[idx]
                scene_name = val_loader.dataset.nusc.get('scene', rec['scene_token'])['name']
                lidar_top_path = val_loader.dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])
                base_name = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0].split('_')[-1] # timestamp
                base_name = scene_name + '_' + base_name # {scene_name}_{timestamp}

                print(f'batch index {batchi:06}_{si:03}')
                
                vector_gt = vectors_gt[si] # [instance] list of dict

                impath = os.path.join(logdir, 'vector_gt')
                if not os.path.exists(impath):
                    os.mkdir(impath)
                imname = os.path.join(impath, f'{base_name}.png')
                print('saving', imname)

                fig = plt.figure(figsize=(4, 2))
                plt.xlim(-30, 30)
                plt.ylim(-15, 15)
                plt.axis('off')

                for vector in vector_gt:
                    pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
                    pts = pts[:pts_num]
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])
                    plt.scatter(x, y, s=1.5, c=colors_plt[line_type])
                    plt.plot(x, y, linewidth=2.0, color=colors_plt[line_type], alpha=0.7)
                plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
                plt.savefig(imname, bbox_inches='tight', pad_inches=0, dpi=400)
                plt.close()

                impath = os.path.join(logdir, 'images')
                if not os.path.exists(impath):
                    os.mkdir(impath)
                imname = os.path.join(impath, f'{base_name}.jpg')
                print('saving', imname)

                fig = plt.figure(figsize=(8, 3))
                for i, (img, intrin, rot, tran, cam) in enumerate(zip(imgs[si], intrins[si], rots[si], trans[si], CAMS)):
                    img = np.array(denormalize_img(img)) # h, w, 3
                    intrin = intrin_scale @ intrin
                    P = get_proj_mat(intrin, rot, tran)
                    ax = fig.add_subplot(2, 3, i+1)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    for coord, confidence, line_type in zip(coords, confidences, line_types):
                        coord = coord * dx + bx # [-30, -15, 30, 15]
                        pts, pts_num = coord, coord.shape[0]
                        zeros = np.zeros((pts_num, 1))
                        ones = np.ones((pts_num, 1))
                        world_coords = np.concatenate([pts, zeros, ones], axis=1).transpose(1, 0)
                        pix_coords = perspective(world_coords, P)
                        x = np.array([pts[0] for pts in pix_coords], dtype='int')
                        y = np.array([pts[1] for pts in pix_coords], dtype='int')
                        for j in range(1, x.shape[0]):
                            img = cv2.line(img, (x[j-1], y[j-1]), (x[j], y[j]), color=tuple([255*c for c in to_rgb(colors_plt[line_type])]), thickness=2)
                    if i > 2:
                        img = cv2.flip(img, 1)
                    img = cv2.resize(img, (1600, 900), interpolation=cv2.INTER_CUBIC)
                    text_size, _ = cv2.getTextSize(cam, cv2.FONT_HERSHEY_COMPLEX, 3, 3)
                    text_w, text_h = text_size
                    cv2.rectangle(img, (0, 0), (0+text_w, 0+text_h), color=(0, 0, 0), thickness=-1)
                    img = cv2.putText(img, cam, (0, 0 + text_h + 1 - 1), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
                    ax.imshow(img)
                    
                plt.subplots_adjust(wspace=0.0, hspace=0.0)
                plt.savefig(imname, bbox_inches='tight', pad_inches=0, dpi=400)
                plt.close()

                # Vector map
                impath = os.path.join(logdir, 'vector_pred_final')
                if not os.path.exists(impath):
                    os.mkdir(impath)
                imname = os.path.join(impath, f'{base_name}.png')
                print('saving', imname)

                fig = plt.figure(figsize=(4, 2))
                plt.xlim(-30, 30)
                plt.ylim(-15, 15)
                plt.axis('off')

                for coord, confidence, line_type in zip(coords, confidences, line_types):
                    coord = coord * dx + bx # [-30, -15, 30, 15]
                    x = np.array([pt[0] for pt in coord])
                    y = np.array([pt[1] for pt in coord])
                    plt.scatter(coord[:, 0], coord[:, 1], 1.5, c=colors_plt[line_type])
                    plt.plot(coord[:, 0], coord[:, 1], linewidth=2.0, color=colors_plt[line_type], alpha=0.7)
                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])
                    # plt.plot(x, y, '-', c=colors_plt[line_type], linewidth=2)
                plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
                plt.savefig(imname, bbox_inches='tight', pad_inches=0, dpi=400)
                plt.close()

                # Instance map
                impath = os.path.join(logdir, 'instance_pred')
                if not os.path.exists(impath):
                    os.mkdir(impath)
                imname = os.path.join(impath, f'{base_name}.png')
                print('saving', imname)

                fig = plt.figure(figsize=(4, 2))
                plt.xlim(-30, 30)
                plt.ylim(-15, 15)
                plt.axis('off')

                for coord in coords:
                    coord = coord * dx + bx # [-30, -15, 30, 15]
                    plt.plot(coord[:, 0], coord[:, 1], linewidth=2)
                plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
                plt.savefig(imname, bbox_inches='tight', pad_inches=0, dpi=400)
                plt.close()
                
                # # Vector instances with grids
                # impath = os.path.join(logdir, 'instance_grid')
                # if not os.path.exists(impath):
                #     os.mkdir(impath)
                # imname = os.path.join(impath, f'eval{batchi:06}_{si:03}.png')
                # print('saving', imname)

                # plt.figure(figsize=(4, 2))
                # plt.xlim(-30, 30)
                # plt.ylim(-15, 15)
                # # plt.axis('off')
                # major_xticks = np.linspace(-30, 30, 51)
                # major_yticks = np.linspace(-15, 15, 26)
                # plt.xticks(major_xticks, fontsize=2)
                # plt.yticks(major_yticks, fontsize=2)
                # plt.grid(True)

                # for coord in coords:
                #     coord = coord * dx + bx # [-30, -15, 30, 15]
                #     plt.plot(coord[:, 0], coord[:, 1], 'o-', linewidth=0.1, markersize=0.5)
                # plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
                # plt.savefig(imname, bbox_inches='tight', dpi=400)
                # plt.close()


def vis_vectormapnet_scene(dataroot, version, model, args):
    data_conf = {
        'image_size': (900, 1600),
        'xbound': args.xbound,
        'ybound': args.ybound
    }

    dataset = VectorMapNetDataset(version=version, dataroot=dataroot, data_conf=data_conf, is_train=False)
    car_img = Image.open('icon/car.png')
    
    # data_conf = {
    #     'num_channels': NUM_CLASSES + 1,
    #     'image_size': args.image_size,
    #     'xbound': args.xbound,
    #     'ybound': args.ybound,
    #     'zbound': args.zbound,
    #     'dbound': args.dbound,
    #     'thickness': args.thickness,
    #     'angle_class': args.angle_class,
    #     'dist_threshold': args.dist_threshold, # 10.0
    #     'cell_size': args.cell_size, # 8
    #     'num_vectors': args.num_vectors, # 100
    #     'pos_freq': args.pos_freq, # 10
    #     'feature_dim': args.feature_dim, # 256
    #     'gnn_layers': args.gnn_layers, # ['self']*7
    #     'sinkhorn_iterations': args.sinkhorn_iterations, # 100
    #     'vertex_threshold': args.vertex_threshold, # 0.015
    #     'match_threshold': args.match_threshold, # 0.1
    # } 

    for idx in tqdm.tqdm(range(dataset.__len__())):
        rec = dataset.nusc.sample[idx]
        scene_name = dataset.nusc.get('scene', rec['scene_token'])['name']
        imgs, trans, rots, intrins, post_trans, post_rots = dataset.get_imgs(rec)
        lidar_data, lidar_mask = dataset.get_lidar(rec)
        car_trans, yaw_pitch_roll = dataset.get_ego_pose(rec)
        vectors = dataset.get_vectors(rec)

        lidar_top_path = dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])
        base_name = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0].split('_')[-1] # timestamp
        base_name = scene_name + '_' + base_name # {scene_name}_{timestamp}

        # imgs: 6, 3, 900, 1600
        # intrins: 6, 3, 3
        # trans: 6, 3, 1
        # rots: 6, 3, 3


def main(args):
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
        'match_threshold': args.match_threshold, # 0.1
    }

    train_loader, val_loader = vectormap_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    norm_layer_dict = {'1d': torch.nn.BatchNorm1d, '2d': torch.nn.BatchNorm2d}
    model = get_model(args.model, data_conf, norm_layer_dict, args.segmentation, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class, args.distance_reg, args.vertex_pred, args.refine)
    model.load_state_dict(torch.load(args.modelf, map_location='cuda:0'), strict=False)
    model.cuda()
    # vis_vector(model, val_loader, args.angle_class, args.logdir)
    vis_segmentation(model, val_loader, args.logdir, args.distance_reg, args.dist_threshold, args.vertex_pred, args.cell_size, args.vertex_threshold)
    # if args.instance_seg and args.direction_pred:
    #     vis_vector(model, val_loader, args.angle_class, args.logdir)
    # if args.model == 'VectorMapNet_cam':
    #     vis_vectormapnet(model, val_loader, args.logdir, data_conf)
        # vis_vectormapnet_scene(args.dataroot, args.version, model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs/match_align_debug_v3')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='./nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='VectorMapNet_cam')
    parser.add_argument("--backbone", type=str, default='efficientnet-b4',
                        choices=['efficientnet-b0', 'efficientnet-b4', 'efficientnet-b7', 'resnet-18', 'resnet-50'])

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
    parser.add_argument('--modelf', type=str, default='./runs/match_align_debug_v3/model_best.pt')

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

    # distance transform config
    parser.add_argument("--distance_reg", action='store_true')
    parser.add_argument("--dist_threshold", type=float, default=10.0)

    # vertex location classification config
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
    parser.add_argument("--vertex_threshold", type=float, default=0.01)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=7)
    parser.add_argument("--sinkhorn_iterations", type=int, default=100)
    parser.add_argument("--match_threshold", type=float, default=0.1)

    args = parser.parse_args()
    main(args)