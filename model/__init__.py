from .hdmapnet import HDMapNet
from .ipm_net import IPMNet
from .lift_splat import LiftSplat
from .pointpillar import PointPillar
from .vectormapnet import VectorMapNet

import torch.nn as nn

def get_model(method, data_conf, norm_layer_dict, segmentation=True, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36, distance_reg=True, vertex_pred=True, refine=False):
    if method == 'lift_splat':
        model = LiftSplat(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_cam':
        model = HDMapNet(data_conf, segmentation=segmentation, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False, distance_reg=distance_reg, vertex_pred=vertex_pred)
    elif method == 'HDMapNet_lidar':
        model = PointPillar(data_conf, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_fusion':
        model = HDMapNet(data_conf, segmentation=segmentation, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True, distance_reg=distance_reg, vertex_pred=vertex_pred)
    elif method == "InstaGraM_cam":
        model = VectorMapNet(data_conf, norm_layer_dict, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False, distance_reg=distance_reg, refine=refine)
    elif method == "InstaGraM_fusion":
        model = VectorMapNet(data_conf, norm_layer_dict, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True, distance_reg=distance_reg, refine=refine)
    else:
        raise NotImplementedError

    return model
