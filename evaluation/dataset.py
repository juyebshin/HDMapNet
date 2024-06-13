import os
import json
import tqdm
import numpy as np

import torch

from data.dataset import HDMapNetDataset, VectorMapDataset
from data.rasterize import rasterize_map
from data.const import NUM_CLASSES, MAP_CLASSES
from nuscenes.utils.splits import create_splits_scenes


class HDMapNetEvalDataset(HDMapNetDataset):
    def __init__(self, version, dataroot, eval_set, result_path, data_conf, max_line_count=300):
        self.eval_set = eval_set
        super(HDMapNetEvalDataset, self).__init__(version, dataroot, data_conf, is_train=False)
        with open(result_path, 'r') as f:
            self.prediction = json.load(f)
        self.max_line_count = max_line_count
        self.thickness = data_conf['thickness']

    def get_scenes(self, version, is_train):
        return create_splits_scenes()[self.eval_set]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        gt_vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])

        gt_map, _ = rasterize_map(gt_vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness)
        if self.prediction['meta']['vector']:
            pred_vectors = next(item for item in self.prediction['results'] if item['sample_token'] == rec['token'])['vectors'] # self.prediction['results'] is list!!
            pred_map, confidence_level = rasterize_map(pred_vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness)
        else:
            pred = next(item for item in self.prediction['results'] if item['sample_token'] == rec['token'])
            pred_map = np.array(pred['map'])
            confidence_level = self.prediction['results'][rec['token']]['confidence_level']

        confidence_level = torch.tensor(confidence_level + [-1] * (self.max_line_count - len(confidence_level)))

        return pred_map, confidence_level, gt_map

class VectorMapEvalDataset(VectorMapDataset):
    def __init__(self, version, dataroot, eval_set, result_path, data_conf, map_ann_file=None):
        self.eval_set = eval_set
        super(VectorMapEvalDataset, self).__init__(version, dataroot, data_conf, is_train=False, map_ann_file=map_ann_file)
        # with open(result_path, 'r') as f:
        #     self.prediction = json.load(f)
        self.result_path = result_path
        self.map_ann_file = map_ann_file
        self.pc_range = [data_conf['xbound'][0], data_conf['ybound'][0], -5.0, data_conf['xbound'][1], data_conf['ybound'][1], 3.0]
    
    def _format_gt(self):
        gt_annos = []
        print('Start to convert gt map format...')
        assert self.map_ann_file is not None
        if (not os.path.exists(self.map_ann_file)):
            dataset_length = len(self.samples)
            for sample_id in tqdm.tqdm(range(dataset_length)):
                rec = self.samples[sample_id]
                gt_vectors = self.get_vectors(rec)
                for vector in gt_vectors:
                    vector['cls_name'] = MAP_CLASSES[vector['type']]
                    # vector['confidence_level'] = 1
                    vector['pts'] = vector['pts'].tolist()
                gt_anno = {}
                gt_anno['sample_token'] = rec['token']
                gt_anno['vectors'] = gt_vectors # gt_vectors: keys(pts, pts_num, cls_name, type, confidence_level)
                gt_annos.append(gt_anno)
                # gt_annos[rec['token']] = gt_vectors # gt_vectors: keys(pts, pts_num, cls_name, type, confidence_level)
            nusc_submissions = {'GTs': gt_annos}
            with open(self.map_ann_file, 'w') as f:
                json.dump(nusc_submissions, f)
        else:
            print(f'{self.map_ann_file} exists, not update.')

    def get_scenes(self, version, is_train):
        return create_splits_scenes()[self.eval_set]
    
    def __len__(self):
        return len(self.samples)
    
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         show=False,
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
                                                     pc_range=self.pc_range,
                                                     show=show)

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
            cls_aps = np.zeros((len(thresholds),NUM_CLASSES))

            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(
                                gen_results,
                                annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=MAP_CLASSES,
                                logger=logger,
                                pc_range=self.pc_range,
                                metric=metric)
                for j in range(NUM_CLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']

            for i, name in enumerate(MAP_CLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['NuscMap_{}/{}_AP'.format(metric,name)] =  cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['NuscMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()

            for i, name in enumerate(MAP_CLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]

        return detail
    
    def evaluate(self,
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
        self._format_gt()

        if isinstance(self.result_path, dict):
            results_dict = dict()
            for name in self.result_path.keys():
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(self.result_path[name], metric=metric)
            results_dict.update(ret_dict)
        elif isinstance(self.result_path, str):
            results_dict = self._evaluate_single(self.result_path, metric=metric, show=show)

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        # if show:
        #     self.show(results, out_dir, pipeline=pipeline)
        return results_dict