import torch
from torch import nn

from .hdmapnet import HDMapNet

class VectorMapNet(nn.Module):
    def __init__(self, data_conf, instance_seg=False, embedded_dim=16, direction_pred=False, direction_dim=36, lidar=False, distance_reg=True, vertex_pred=True) -> None:
        super(VectorMapNet, self).__init__()

        self.hdmapnet = HDMapNet(data_conf, instance_seg, embedded_dim, direction_pred, direction_dim, lidar, distance_reg, vertex_pred)

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        semantic, distance, vertex, embedding, direction = self.hdmapnet(img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll)
        # semantic, embedding, direction are not used

        return semantic, distance, vertex, embedding, direction