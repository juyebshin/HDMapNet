import os
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
import cv2

from data.dataset import VectorMapDataset
from data.const import CAMS
from data.utils import get_proj_mat, perspective
from data.image import denormalize_img


def vis_label(dataroot, version, xbound, ybound, dbound, sample_dist, is_train):
    data_conf = {
        'image_size': (256, 704),
        'xbound': xbound,
        'ybound': ybound,
        'dbound': dbound,
        'sample_dist': sample_dist, # 1.5
        'thickness': 5,
        'angle_class': 36,
        'dist_threshold': 10, # 10.0
        'cell_size': 8, # 8
        'pv_seg': True,
        'pv_seg_classes': 1,
        'feat_downsample': 16,
        'depth_gt': True,
    }

    color_map = np.random.randint(0, 256, (256, 3))
    color_map[0] = np.array([0, 0, 0])
    colors_plt = ['tab:red', 'tab:blue', 'tab:green', 'k']

    semantic_color = np.array([
        [255, 0, 0], # line
        [0, 0, 255], # ped_crossing
        [0, 255, 0], # contour
        [0, 0, 0], # background
        ])

    dataset = VectorMapDataset(version=version, dataroot=dataroot, data_conf=data_conf, is_train=is_train)
    gt_path = os.path.join(dataroot, 'GT')
    if not os.path.exists(gt_path):
        os.mkdir(gt_path)
        
    intrin_scale = np.eye(3)
    intrin_scale[0, 0] *= data_conf['image_size'][1] / 1600.
    intrin_scale[1, 1] *= data_conf['image_size'][0] / 900.

    car_img = Image.open('icon/car.png')
    num_vectors_list = []
    for idx in tqdm.tqdm(range(dataset.__len__())):
        rec = dataset.nusc.sample[idx]
        # imgs, trans, rots, intrins, post_trans, post_rots = dataset.get_imgs(rec)
        # vectors = dataset.get_vectors(rec)
        imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, \
        car_trans, yaw_pitch_roll, semantic_masks, instance_masks, distance_masks, \
        vertex_masks, pv_semantic_masks, depth_maps, vectors = dataset[idx]

        lidar_top_path = dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])

        base_path = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0]
        base_path = os.path.join(gt_path, base_path)

        if not os.path.exists(base_path):
            os.mkdir(base_path)
        plt.figure()
        plt.xlim(xbound[0], xbound[1])
        plt.ylim(ybound[0], ybound[1])
        plt.axis('off')
        num_vectors = 0
        for vector in vectors:
            pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
            pts = pts[:pts_num]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])
            plt.scatter(x, y, s=1.5, c=colors_plt[line_type])
            plt.plot(x, y, linewidth=2.0, color=colors_plt[line_type], alpha=0.7)
            num_vectors += pts_num
        num_vectors_list.append([base_path, num_vectors])

        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])

        map_path = os.path.join(base_path, 'MAP.png')
        plt.savefig(map_path, bbox_inches='tight', pad_inches=0, dpi=1200)
        plt.close()

        # major_xticks = np.linspace(int(xbound[0]), int(xbound[1]), int((xbound[1]-xbound[0])/(8*xbound[2])+1))
        # major_yticks = np.linspace(int(ybound[0]), int(ybound[1]), int((ybound[1]-ybound[0])/(8*ybound[2])+1))
        # plt.figure()
        # plt.xlim(xbound[0], xbound[1])
        # plt.ylim(ybound[0], ybound[1])
        # plt.axis('off')
        # plt.xticks(major_xticks, fontsize=2)
        # plt.yticks(major_yticks, fontsize=2)
        # plt.grid(True)
        
        # for vector in vectors:
        #     pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
        #     pts = pts[:pts_num]
        #     x = np.array([pt[0] for pt in pts])
        #     y = np.array([pt[1] for pt in pts])
        #     plt.scatter(x, y, s=0.1, c=colors_plt[line_type])

        # plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])

        # map_path = os.path.join(base_path, 'VERTEX.png')
        # plt.savefig(map_path, bbox_inches='tight', pad_inches=0, dpi=1200)
        # plt.close()
        
        gt_depth_surround = np.zeros((depth_maps.shape[1]*2, depth_maps.shape[2]*3, 3), np.uint8)

        for img, intrin, rot, tran, depth_map, cam in zip(imgs, intrins, rots, trans, depth_maps, CAMS):
            img = denormalize_img(img)
            intrin = intrin_scale @ intrin.numpy()
            P = get_proj_mat(intrin, rot, tran)
            plt.figure()
            fig = plt.imshow(img)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.xlim(0, data_conf['image_size'][1])
            plt.ylim(data_conf['image_size'][0], 0)
            plt.axis('off')
            for vector in vectors:
                pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
                pts = pts[:pts_num]
                zeros = np.zeros((pts_num, 1))
                ones = np.ones((pts_num, 1))
                world_coords = np.concatenate([pts, zeros, ones], axis=1).transpose(1, 0)
                pix_coords = perspective(world_coords, P)
                x = np.array([pts[0] for pts in pix_coords])
                y = np.array([pts[1] for pts in pix_coords])
                # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy',
                #         angles='xy', scale=1, color=colors_plt[line_type])
                plt.plot(x, y, linewidth=2.0, color=colors_plt[line_type], alpha=0.7)

            cam_path = os.path.join(base_path, f'{cam}.png')
            plt.savefig(cam_path, bbox_inches='tight', pad_inches=0, dpi=400)
            plt.close()
            
            pv_semantic_mask = pv_semantic_masks[CAMS.index(cam)] # (num_classes, h, w)
            pv_semantic_mask = pv_semantic_mask.numpy().astype('uint8') * 255
            pv_bg_mask = np.zeros((1, *pv_semantic_mask.shape[1:]), np.uint8) # (1, h, w)
            pv_bg_mask[:, pv_semantic_mask.max(0) == 0] = 255
            pv_semantic_mask = np.concatenate([pv_semantic_mask, pv_bg_mask], axis=0) # (num_classes+1, h, w)
            pv_semantic_mask = np.argmax(pv_semantic_mask, axis=0) # (h, w)
            pv_semantic_color_mask = semantic_color[pv_semantic_mask].astype('uint8')
            plt.figure()
            plt.imshow(pv_semantic_color_mask)
            plt.axis('off')

            pv_cam_path = os.path.join(base_path, f'pv_mask_{cam}.png')
            plt.savefig(pv_cam_path, bbox_inches='tight', pad_inches=0, dpi=400)
            plt.close()
            
            image = np.asarray(img)
            gt_depth_image = depth_map.numpy()
            gt_depth_image = np.expand_dims(gt_depth_image,2).repeat(3,2)
            
            #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
            im_color=cv2.applyColorMap(cv2.convertScaleAbs(gt_depth_image,alpha=15),cv2.COLORMAP_JET)
            #convert to mat png
            image[gt_depth_image>0] = im_color[gt_depth_image>0]
            im=Image.fromarray(np.uint8(image))
            #save image
            gt_depth_path = os.path.join(base_path, f'gt_depth_{cam}.png')
            im.save(gt_depth_path)
            
            idx = CAMS.index(cam)
            gt_depth_surround[image.shape[0]*(idx//3):image.shape[0]*(idx//3)+image.shape[0], 
                              image.shape[1]*(idx%3):image.shape[1]*(idx%3)+image.shape[1]] = image
        gt_depth_surround_path = os.path.join(base_path, 'gt_depth_surround.png')
        Image.fromarray(np.uint8(gt_depth_surround)).save(gt_depth_surround_path)
    
    prefix = 'train' if is_train else 'val'
    if xbound[1] > 30:
        prefix += '_long_range'
    with open(f'num_vectors_{prefix}.csv', 'w') as f:
        print('saving number of vectors list to csv...')
        write = csv.writer(f)
        write.writerows(num_vectors_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local HD Map Demo.')
    parser.add_argument('dataroot', nargs='?', type=str, default='./nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--dbound", nargs=3, type=float, default=[1.0, 35.0, 0.5])
    parser.add_argument("--sample_dist", type=float, default=1.5)
    parser.add_argument("--depth-gt", action='store_true')
    parser.add_argument("--is_train", action='store_true')
    args = parser.parse_args()

    vis_label(args.dataroot, args.version, args.xbound, args.ybound, args.dbound, args.sample_dist, args.is_train)
