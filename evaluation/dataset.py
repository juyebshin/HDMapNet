import json
import numpy as np

import torch

from data.dataset import HDMapNetDataset
from data.rasterize import rasterize_map
from data.const import NUM_CLASSES
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
            pred_vectors = self.prediction['results'][rec['token']]
            pred_map, confidence_level = rasterize_map(pred_vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness)
        else:
            pred_map = np.array(self.prediction['results'][rec['token']]['map'])
            confidence_level = self.prediction['results'][rec['token']]['confidence_level']

        confidence_level = torch.tensor(confidence_level + [-1] * (self.max_line_count - len(confidence_level)))
        # self.visualize_vec(rec, gt_vectors, pred_vectors)
        return pred_map, confidence_level, gt_map

    def visualize_vec(self, rec, gt_vectors, pred_vectors):
        from matplotlib import pyplot as plt
        
        token = rec['token']
        val_ls = []
        with open('/home/user/data/hyeonjun/VectorLoc/samps/sample_val_rand.json', 'r') as f:
            val_ls = json.load(f)

        if token in val_ls:
            fig, ax = plt.subplots(1,2, figsize=(13.0, 7.2))
            ax[0].cla()
            ax[1].cla()
            # color_map = ['gray', 'red', 'orange', 'blue', 'green', 'magenta', 'cyan']
            color_map = ['gray', 'orange', 'blue', 'green', 'magenta', 'cyan']
            color_map = np.array(color_map)
            for i, ins in enumerate(gt_vectors):
                ins_pts = np.array(ins['pts'], dtype=np.float32)
                ax[0].scatter(ins_pts[:,1], ins_pts[:,0], color=color_map[ins['type']+1], s=30)
                ax[0].plot(ins_pts[:,1], ins_pts[:,0], color=color_map[ins['type']+1], linewidth=7, alpha=0.7)
            for i, ins in enumerate(pred_vectors):
                ins_pts = np.array(ins['pts'], dtype=np.float32)
                ax[1].scatter(ins_pts[:,1], ins_pts[:,0], color=color_map[ins['type']+1], s=30)
                ax[1].plot(ins_pts[:,1], ins_pts[:,0], color=color_map[ins['type']+1], linewidth=7, alpha=0.7)
            ax[0].set_ylim(-30,30)
            ax[0].set_xlim(-30,30)
            ax[0].set_aspect('equal')
            ax[1].set_ylim(-30,30)
            ax[1].set_xlim(-30,30)
            ax[1].set_aspect('equal')
            ax[0].set_title('gt')
            ax[1].set_title('pred')
            title = f'{(val_ls.index(token)):04}_{token}'
            fig.suptitle(title)
            fig.savefig(f'tmp/{title}.png')
            plt.close(fig)