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
        lidar_data = get_lidar_data(self.nusc, rec, nsweeps=3, min_distance=2.2)
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
            trans.append(torch.Tensor(sens['translation'])) # tensor.Size([3])
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)) # [3, 3]
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
    def __init__(self, version, dataroot, data_conf, is_train, map_ann_file=None):
        super(VectorMapDataset, self).__init__(version, dataroot, data_conf, is_train)
        self.thickness = data_conf['thickness']
        self.angle_class = data_conf['angle_class']
        self.dist_threshold = data_conf['dist_threshold']
        self.cell_size = data_conf['cell_size']

    def get_vector_map(self, rec):
        vectors = self.get_vectors(rec)
        instance_masks, _, _, distance_masks, vertex_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness, self.angle_class, self.cell_size)
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
        instance_masks = instance_masks.sum(0)
        # obtain normalized DT [0.0, 1.0], truncated by 10
        distance_masks = get_distance_transform(distance_masks, self.dist_threshold)
        return semantic_masks, instance_masks, torch.tensor(distance_masks, dtype=torch.float32), vertex_masks.type(torch.bool), vectors
        
    def __getitem__(self, idx):
        rec = self.samples[idx]
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        semantic_masks, instance_masks, distance_masks, vertex_masks, vectors = self.get_vector_map(rec)
        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, distance_masks, vertex_masks, vectors

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        assert self.map_ann_file is not None
        pred_annos = []
        mapped_class_names = self.MAPCLASSES
        # import pdb;pdb.set_trace()
        print('Start to convert map detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            pred_anno = {}
            vecs = output_to_vecs(det)
            sample_token = self.data_infos[sample_id]['token']
            pred_anno['sample_token'] = sample_token
            pred_vec_list=[]
            for i, vec in enumerate(vecs): # for num_vec
                name = mapped_class_names[vec['label']]
                anno = dict(
                    pts=vec['pts'],
                    pts_num=len(vec['pts']),
                    cls_name=name,
                    type=vec['label'],
                    confidence_level=vec['score'])
                pred_vec_list.append(anno)

            pred_anno['vectors'] = pred_vec_list
            pred_annos.append(pred_anno)


        if not os.path.exists(self.map_ann_file):
            self._format_gt()
        else:
            print(f'{self.map_ann_file} exist, not update')

        nusc_submissions = {
            'meta': self.modality,
            'results': pred_annos,

        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'nuscmap_results.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
    
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='chamfer'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        # import eval_map
        # import format_res_gt_by_classes
        from .mean_ap import eval_map
        from .mean_ap import format_res_gt_by_classes
        result_path = os.path.abspath(result_path)
        detail = dict()

        print('Formating results & gts by classes')
        with open(result_path, 'r') as f:
            pred_results = json.load(f)
        gen_results = pred_results['results']
        with open(self.map_ann_file, 'r') as ann_f:
            gt_anns = json.load(ann_f)
        annotations = gt_anns['GTs']
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     gen_results,
                                                     annotations,
                                                     cls_names=MAP_CLASSES,
                                                     eval_use_same_gt_sample_num_flag=True,
                                                     pc_range=self.pc_range)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        for metric in metrics:
            print('-*'*10+f'use metric:{metric}'+'-*'*10)

            if metric == 'chamfer':
                thresholds = [0.5,1.0,1.5]
            elif metric == 'iou':
                thresholds= np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds),self.NUM_MAPCLASSES))

            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(
                                gen_results,
                                annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=self.MAPCLASSES,
                                logger=logger,
                                num_pred_pts_per_instance=self.fixed_num,
                                pc_range=self.pc_range,
                                metric=metric)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']

            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['NuscMap_{}/{}_AP'.format(metric,name)] =  cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['NuscMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()

            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]

        return detail
    
    def evaluate(self,
                 results,
                 metric='chamfer',
                 logger=None,
                 show=False,
                 out_dir=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        pass
    
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
