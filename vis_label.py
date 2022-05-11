import os
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv

from data.dataset import HDMapNetDataset, CAMS
from data.utils import get_proj_mat, perspective
from data.image import denormalize_img


def vis_label(dataroot, version, xbound, ybound):
    data_conf = {
        'image_size': (900, 1600),
        'xbound': xbound,
        'ybound': ybound
    }

    color_map = np.random.randint(0, 256, (256, 3))
    color_map[0] = np.array([0, 0, 0])
    colors_plt = ['r', 'b', 'g']

    dataset = HDMapNetDataset(version=version, dataroot=dataroot, data_conf=data_conf, is_train=True)
    gt_path = os.path.join(dataroot, 'samples', 'GT')
    if not os.path.exists(gt_path):
        os.mkdir(gt_path)

    car_img = Image.open('icon/car.png')
    num_vectors_list = []
    for idx in tqdm.tqdm(range(dataset.__len__())):
        rec = dataset.nusc.sample[idx]
        imgs, trans, rots, intrins, post_trans, post_rots = dataset.get_imgs(rec)
        vectors = dataset.get_vectors(rec)

        lidar_top_path = dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])

        base_path = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0]
        base_path = os.path.join(gt_path, base_path)

        if not os.path.exists(base_path):
            os.mkdir(base_path)

        major_xticks = np.linspace(-30, 30, 51)
        major_yticks = np.linspace(-15, 15, 26)
        plt.figure(figsize=(4, 2))
        plt.xlim(-30, 30)
        plt.ylim(-15, 15)
        plt.axis('off')
        # plt.xticks(major_xticks, fontsize=2)
        # plt.yticks(major_yticks, fontsize=2)
        # plt.grid(True)
        num_vectors = 0
        for vector in vectors:
            pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
            pts = pts[:pts_num]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])
            # plt.scatter(x, y, s=0.1, c=colors_plt[line_type])
            num_vectors += pts_num
        num_vectors_list.append([base_path, num_vectors])

        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])

        map_path = os.path.join(base_path, 'MAP.png')
        # plt.savefig(map_path, bbox_inches='tight', dpi=400)
        plt.close()

        # for img, intrin, rot, tran, cam in zip(imgs, intrins, rots, trans, CAMS):
        #     img = denormalize_img(img)
        #     P = get_proj_mat(intrin, rot, tran)
        #     plt.figure(figsize=(9, 16))
        #     fig = plt.imshow(img)
        #     fig.axes.get_xaxis().set_visible(False)
        #     fig.axes.get_yaxis().set_visible(False)
        #     plt.xlim(1600, 0)
        #     plt.ylim(900, 0)
        #     plt.axis('off')
        #     for vector in vectors:
        #         pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
        #         pts = pts[:pts_num]
        #         zeros = np.zeros((pts_num, 1))
        #         ones = np.ones((pts_num, 1))
        #         world_coords = np.concatenate([pts, zeros, ones], axis=1).transpose(1, 0)
        #         pix_coords = perspective(world_coords, P)
        #         x = np.array([pts[0] for pts in pix_coords])
        #         y = np.array([pts[1] for pts in pix_coords])
        #         plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy',
        #                 angles='xy', scale=1, color=colors_plt[line_type])

        #     cam_path = os.path.join(base_path, f'{cam}.png')
        #     plt.savefig(cam_path, bbox_inches='tight', pad_inches=0, dpi=400)
        #     plt.close()
    
    with open('num_vectors_train.csv', 'w') as f:
        print('saving number of vectors list to csv...')
        write = csv.writer(f)
        write.writerows(num_vectors_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local HD Map Demo.')
    parser.add_argument('dataroot', nargs='?', type=str, default='/home/user/data/Dataset/nuscenes/v1.0-trainval/')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    args = parser.parse_args()

    vis_label(args.dataroot, args.version, args.xbound, args.ybound)
