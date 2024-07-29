import os
import numpy as np

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from torch.utils.data import DataLoader, DistributedSampler

from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from data.rasterize import preprocess_map
from data.utils import get_proj_mat
from .const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W, MAP_CLASSES
from .vector_map import VectorizedLocalMap
from .lidar import get_lidar_data
from .image import normalize_img, img_transform
from .utils import label_onehot_encoding, get_distance_transform
from model.voxel import pad_or_trim_to_np


class HDMapNetDataset(Dataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(HDMapNetDataset, self).__init__()
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0] # 30.0
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0] # 60.0
        canvas_h = int(patch_h / data_conf['ybound'][2])
        canvas_w = int(patch_w / data_conf['xbound'][2])
        self.is_train = is_train
        self.data_conf = data_conf
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = NuScenes(version=version, dataroot=dataroot)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size, sample_dist=data_conf['sample_dist'])
        self.scenes = self.get_scenes(version, is_train)
        self.samples = self.get_samples()

    def __len__(self):
        return len(self.samples)

    def get_scenes(self, version, is_train):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[version][is_train]

        return create_splits_scenes()[split]

    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        
        # # keyframe blob only
        # samples = []
        # for scene in self.nusc.scene:

        #     # Ignore scenes which don't belong to the current split
        #     if scene['name'] not in self.scenes:
        #         continue

        #     # iterate over samples
        #     for sample in self.iterate_samples(scene['first_sample_token']):

        #         samples.append(sample)

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def iterate_samples(self, start_token):
        sample_token = start_token
        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            yield sample
            sample_token = sample['next']

    def get_lidar(self, rec):
        '''Get lidar data and mask for a sample projected to ego frame.'''
        lidar_data = get_lidar_data(self.nusc, rec, nsweeps=5, min_distance=2.2)
        lidar_data = lidar_data.transpose(1, 0)
        num_points = lidar_data.shape[0]
        lidar_data = pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0
        return lidar_data, lidar_mask

    def get_ego_pose(self, rec):
        sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        car_trans = ego_pose['translation']
        pos_rotation = Quaternion(ego_pose['rotation'])
        yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)

    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size'] # 128, 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H) # 0.22, 0.142
        resize_dims = (fW, fH) # 128, 352
        return resize, resize_dims

    # def sample_augmentation(self):
    #     self.data_conf['resize_lim'] = (0.193, 0.225)
    #     self.data_conf['bot_pct_lim'] = (0.0, 0.22)
    #     self.data_conf['rand_flip'] = True
    #     self.data_conf['rot_lim'] = (-5.4, -5.4)

    #     fH, fW = self.data_conf['image_size']
    #     if self.is_train:
    #         resize = np.random.uniform(*self.data_conf['resize_lim'])
    #         resize_dims = (int(IMG_ORIGIN_W*resize), int(IMG_ORIGIN_H*resize))
    #         newW, newH = resize_dims
    #         crop_h = int((1 - np.random.uniform(*self.data_conf['bot_pct_lim']))*newH) - fH
    #         crop_w = int(np.random.uniform(0, max(0, newW - fW)))
    #         crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    #         flip = False
    #         if self.data_conf['rand_flip'] and np.random.choice([0, 1]):
    #             flip = True
    #         rotate = np.random.uniform(*self.data_conf['rot_lim'])
    #     else:
    #         resize = max(fH/IMG_ORIGIN_H, fW/IMG_ORIGIN_W)
    #         resize_dims = (int(IMG_ORIGIN_W*resize), int(IMG_ORIGIN_H*resize))
    #         newW, newH = resize_dims
    #         crop_h = int((1 - np.mean(self.data_conf['bot_pct_lim']))*newH) - fH
    #         crop_w = int(max(0, newW - fW) / 2)
    #         crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    #         flip = False
    #         rotate = 0
    #     return resize, resize_dims, crop, flip, rotate


    def get_imgs(self, rec):
        imgs = []
        trans = []
        rots = []
        intrins = []
        post_trans = []
        post_rots = []

        for cam in CAMS:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)

            resize, resize_dims = self.sample_augmentation() # [0.22, 0.142], [128, 352]
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)
            # resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            # img, post_rot, post_tran = img_transform(img, resize, resize_dims, crop, flip, rotate)

            img = normalize_img(img)
            post_trans.append(post_tran)
            post_rots.append(post_rot)
            imgs.append(img)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            trans.append(torch.Tensor(sens['translation'])) # tensor.Size([3]) cam2ego translation
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)) # [3, 3] cam2ego rotation
            intrins.append(torch.Tensor(sens['camera_intrinsic']))
        return torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(post_trans), torch.stack(post_rots)

    def get_vectors(self, rec):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        return vectors

    def __getitem__(self, idx):
        rec = self.samples[idx]
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        vectors = self.get_vectors(rec)

        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, vectors


class HDMapNetSemanticDataset(HDMapNetDataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(HDMapNetSemanticDataset, self).__init__(version, dataroot, data_conf, is_train)
        self.thickness = data_conf['thickness']
        self.angle_class = data_conf['angle_class']
        self.dist_threshold = data_conf['dist_threshold']
        self.cell_size = data_conf['cell_size']

    def get_semantic_map(self, rec):
        vectors = self.get_vectors(rec)
        instance_masks, forward_masks, backward_masks, distance_masks, vertex_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness, self.angle_class)
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
        instance_masks = instance_masks.sum(0)
        forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
        backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
        direction_masks = forward_oh_masks + backward_oh_masks
        direction_masks = direction_masks / direction_masks.sum(0)
        # obtain normalized DT [0.0, 1.0], truncated by 10
        distance_masks = get_distance_transform(distance_masks, self.dist_threshold)
        return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks, torch.tensor(distance_masks, dtype=torch.float32), vertex_masks.type(torch.bool)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        semantic_masks, instance_masks, _, _, direction_masks, distance_masks, vertex_masks = self.get_semantic_map(rec)
        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_masks, distance_masks, vertex_masks


def semantic_dataset(version, dataroot, data_conf, bsz, nworkers):
    train_dataset = HDMapNetSemanticDataset(version, dataroot, data_conf, is_train=True)
    val_dataset = HDMapNetSemanticDataset(version, dataroot, data_conf, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_loader, val_loader

class VectorMapDataset(HDMapNetDataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(VectorMapDataset, self).__init__(version, dataroot, data_conf, is_train)
        self.thickness = data_conf['thickness']
        self.angle_class = data_conf['angle_class']
        self.dist_threshold = data_conf['dist_threshold']
        self.cell_size = data_conf['cell_size']
        
        self.pv_seg = data_conf.get('pv_seg', False)
        self.pv_seg_classes = data_conf.get('pv_seg_classes', 1)
        self.feat_downsample = data_conf.get('feat_downsample', 1)
        
        self.depth_gt = data_conf.get('depth_gt', False)
        self.dbound = data_conf['dbound']
        self.depth_downsample = data_conf.get('depth_downsample', 1)

    def points2depthmap(self, points, height, width):
        height, width = height // self.depth_downsample, width // self.depth_downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.depth_downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.dbound[1]) & (
                    depth >= self.dbound[0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map
    
    def get_lidar_depth(self, 
                        rec, 
                        lidar_data, 
                        lidar_mask,
                        imgs,
                        trans, # cam2ego [6, 3]
                        rots, # cam2ego [6, 3, 3]
                        intrins,
                        post_trans, # [6, 3]
                        post_rots, # [6, 3, 3]
                        ):
        points_lidar = torch.tensor(lidar_data[lidar_mask > 0, :3]).to(torch.float32)
        sd_rec = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        # lidarego to global transform only, since lidar is already projected to lidarego
        lidarego2global = np.eye(4, dtype=np.float32)
        lidarego2global[:3, :3] = Quaternion(pose_record['rotation']).rotation_matrix
        lidarego2global[:3, 3] = pose_record['translation']
        lidarego2global = torch.from_numpy(lidarego2global)
        depth_map_list = []
        
        for cid in range(len(imgs)):
            cam = CAMS[cid]
            cam_sd_rec = self.nusc.get('sample_data', rec['data'][cam])
            cam_cs_record = self.nusc.get('calibrated_sensor',
                                cam_sd_rec['calibrated_sensor_token'])
            cam_pose_record = self.nusc.get('ego_pose', cam_sd_rec['ego_pose_token'])
            # camera to ego transform
            cam2camego = np.eye(4).astype(np.float32)
            cam2camego[:3, :3] = Quaternion(cam_cs_record['rotation']).rotation_matrix
            cam2camego[:3, 3] = cam_cs_record['translation']
            cam2camego = torch.tensor(cam2camego) # 4x4
            # camego to global transform
            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(cam_pose_record['rotation']).rotation_matrix
            camego2global[:3, 3] = cam_pose_record['translation']
            camego2global = torch.from_numpy(camego2global)
            # intrinsic
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = intrins[cid]
            cam2img = torch.tensor(camera_intrinsics).to(torch.float32)
            
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global)
            lidar2img = cam2img.matmul(lidar2cam)

            points_img = points_lidar.matmul(
                lidar2img[:3, :3].T.to(torch.float)) + lidar2img[:3, 3].to(torch.float).unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            # points_pad = torch.ones((points_lidar.shape[0], 1)).to(points_lidar.dtype)
            # points_pad = torch.cat((points_lidar, points_pad), -1)
            # points_img_homo = points_pad.matmul(
            #     lidar2img.T.to(torch.float))
            # points_img_homo = points_img_homo / points_img_homo[:, -1].unsqueeze(-1)
            # points_img = points_img_homo[:, :-1]
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        
        return depth_map

    def get_vector_map(self, rec, img_shape=None, lidar2feat=None):
        vectors = self.get_vectors(rec)
        instance_masks, _, _, distance_masks, vertex_masks, pv_semantic_masks = preprocess_map(
            vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness, self.angle_class, self.cell_size,
            self.pv_seg_classes, img_shape, lidar2feat)
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
        instance_masks = instance_masks.sum(0)
        # obtain normalized DT [0.0, 1.0], truncated by 10
        distance_masks = get_distance_transform(distance_masks, self.dist_threshold)
        return semantic_masks, instance_masks, torch.tensor(distance_masks, dtype=torch.float32), vertex_masks.type(torch.bool), vectors, pv_semantic_masks
        
    def __getitem__(self, idx):
        rec = self.samples[idx]
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        # lidar depth map
        depth_map = torch.zeros((imgs.shape[0])).to(torch.float)
        if self.depth_gt:
            depth_map = self.get_lidar_depth(rec, lidar_data, lidar_mask, imgs, trans, rots, intrins, post_trans, post_rots)
        # cam projection matrix
        lidar2feat = []
        for i, cam in enumerate(CAMS):
            post_intrin = post_rots[i] @ intrins[i]
            l2i = get_proj_mat(post_intrin, rots[i], trans[i])
            scale_factor = np.eye(4)
            scale_factor[0, 0] *= 1/self.feat_downsample
            scale_factor[1, 1] *= 1/self.feat_downsample
            lidar2feat.append(scale_factor @ l2i)
        img_shape = np.array(imgs.shape[-2:]) // self.feat_downsample
        semantic_masks, instance_masks, distance_masks, vertex_masks, vectors, pv_semantic_masks = self.get_vector_map(rec, img_shape, lidar2feat)
        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, distance_masks, vertex_masks, pv_semantic_masks, depth_map, vectors
    
def vectormap_dataset(version, dataroot, data_conf, bsz, nworkers, distributed=False):
    train_dataset = VectorMapDataset(version, dataroot, data_conf, is_train=True)
    val_dataset = VectorMapDataset(version, dataroot, data_conf, is_train=False)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, bsz, drop_last=True)

    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=nworkers, collate_fn=collate_vectors)
    val_loader = DataLoader(val_dataset, batch_size=bsz, sampler=val_sampler, drop_last=False, num_workers=nworkers, collate_fn=collate_vectors)
    return train_loader, val_loader

def collate_vectors(batch):
    vectors_list = []
    batch_list = []
    # batch: list of 'batch_size' tuple elements
    for *_, vectors in batch:
        # vectors: list of dict
        vectors_list.append(vectors)
        batch_list.append(tuple(_))
    
    batch = default_collate(batch_list)
    # vectors_list: 'batch_size' list of list of dict
    # batch: list
    return tuple([*batch, vectors_list])

if __name__ == '__main__':
    data_conf = {
        'image_size': (900, 1600),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }

    dataset = HDMapNetSemanticDataset(version='v1.0-mini', dataroot='dataset/nuScenes', data_conf=data_conf, is_train=False)
    for idx in range(dataset.__len__()):
        imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_mask = dataset.__getitem__(idx)
