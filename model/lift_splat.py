"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F

from data.utils import gen_dx_bx
from .base import CamEncode, BevEncode, InstaGraM
from .pointpillar import PointPillarEncoder


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class LiftSplat(nn.Module):
    def __init__(self, data_conf, segmentation, instance_seg, embedded_dim, direction_pred, distance_reg, vertex_pred, norm_layer_dict, lidar=False, refine=False):
        super(LiftSplat, self).__init__()
        self.data_conf = data_conf

        dx, bx, nx = gen_dx_bx(data_conf['xbound'],
                                data_conf['ybound'],
                                data_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.pv_seg = data_conf['pv_seg']

        self.downsample = 16
        self.camC = 64 if 'efficientnet' in data_conf['backbone'] else 256        
        self.lidar = lidar
        self.frustum = self.create_frustum()
        # D x H/downsample x D/downsample x 3
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.camC, data_conf['backbone'], norm_layer_dict['2d'], data_conf['pv_seg'], data_conf['pv_seg_classes']) # self.downsample, 
        self.depthnet = nn.Conv2d(self.camC, self.D + self.camC, kernel_size=1, padding=0)
        if lidar:
            self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
            self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], norm_layer=norm_layer_dict['2d'], segmentation=segmentation, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, distance_reg=distance_reg, vertex_pred=vertex_pred, cell_size=self.data_conf['cell_size'])
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], norm_layer=norm_layer_dict['2d'], segmentation=segmentation, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, distance_reg=distance_reg, vertex_pred=vertex_pred, cell_size=self.data_conf['cell_size'])
        self.head = InstaGraM(data_conf, norm_layer_dict['1d'], distance_reg, refine)
        
        if data_conf['pv_seg']:
            self.pv_seg_head = nn.Sequential(
                    nn.Conv2d(self.camC, self.camC, kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.camC, data_conf['pv_seg_classes'], kernel_size=1, padding=0)
                )

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous() 
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        # 把gt_depth做feat_down_sample倍数的采样
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        # 因为深度很稀疏，大部分的点都是0，所以把0变成10000，下一步取-1维度上的最小就是深度的值
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        gt_depths = (
            gt_depths -
            (self.data_conf['dbound'][0] - 
             self.data_conf['dbound'][2])) / self.data_conf['dbound'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()
    
    def get_depth_loss(self, depth_labels, depth_preds):
        # import pdb;pdb.set_trace()
        if depth_preds is None:
            return 0
        
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D) # B*N, D, h, w -> B*N*h*w, D
        # fg_mask = torch.max(depth_labels, dim=1).values > 0.0 # 只计算有深度的前景的深度loss
        # import pdb;pdb.set_trace()
        fg_mask = depth_labels > 0.0 # 只计算有深度的前景的深度loss
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        # if depth_loss <= 0.:
        #     import pdb;pdb.set_trace()
        return depth_loss

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_conf['image_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.data_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # *undo* post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3) # 4, 6, 41, 8, 22, 3
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points # B, N(=6), D(=41), H(=8), W(=22), 3

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x) # B*N, C, h, w
        depth_x = self.depthnet(x) # B*N, D+C, h, w
        depth = depth_x[:, : self.D].softmax(dim=1)
        depth_x = depth.unsqueeze(1) * depth_x[:, self.D:(self.D + self.camC)].unsqueeze(2) # [B*N, 1, 41, 8, 22] * [B*N, 64, 1, 8, 22]
        depth_x = depth_x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample) # B, N, C, D, h, w
        depth_x = depth_x.permute(0, 1, 3, 4, 5, 2).contiguous() # B, N, D, h, w, C

        return x, depth_x, depth # B, N, D(=41), h, w, C(=64)

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape # B, N(=6), D(=41), h, w, C(=64)
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        # B x N x D x H/downsample x W/downsample x 3 // geom_feats: B, N(=6), D(=41), h, w, 3
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # x, y, z, b

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final # B, C(=64), Y, X

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # B x N x D x H/downsample x W/downsample x 3: (x,y,z) locations (in the ego frame)
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # B x N x D x H/downsample x W/downsample x C: cam feats
        x, depth_x, depth = self.get_cam_feats(x)

        depth_x = self.voxel_pooling(geom, depth_x)

        return x, depth_x, depth

    def forward(self, x, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        # imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(), post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(), lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda()
        x, depth_x, depth = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        if self.lidar:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            depth_x = torch.cat([depth_x, lidar_feature], dim=1)
        x_seg, x_dt, x_vertex, x_embedded, x_direction = self.bevencode(depth_x)
        semantic, distance, vertex, matches, positions, masks = self.head(x_dt, x_vertex)
        if self.pv_seg:
            pv_seg = self.pv_seg_head(x)
            return semantic, distance, vertex, matches, positions, masks, x_embedded, x_direction, depth, pv_seg
        return semantic, distance, vertex, matches, positions, masks, x_embedded, x_direction, depth
