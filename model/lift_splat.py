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
from mmcv.cnn import build_conv_layer
from torchvision.models.resnet import BasicBlock

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 aspp_mid_channels=-1,
                 only_depth=False):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.only_depth = only_depth or context_channels == 0
        if not self.only_depth:
            self.context_conv = nn.Conv2d(
                mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
            self.context_mlp = Mlp(22, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.bn = nn.BatchNorm1d(22)
        self.depth_mlp = Mlp(22, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware

        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            if aspp_mid_channels<0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        if not self.only_depth:
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context = self.context_se(x, context_se)
            context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(depth)
        if not self.only_depth:
            return torch.cat([depth, context], dim=1)
        else:
            return depth

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
        self.camC = 256# 64 if 'efficientnet' in data_conf['backbone'] else 256        
        self.lidar = lidar
        self.frustum = self.create_frustum()
        # D x H/downsample x D/downsample x 3
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.camC, data_conf['backbone'], norm_layer_dict['2d'], data_conf['pv_seg'], data_conf['pv_seg_classes']) # self.downsample, 
        # self.depthnet = nn.Conv2d(self.camC, self.D + self.camC, kernel_size=1, padding=0)
        self.depthnet = DepthNet(self.camC, self.camC, self.camC, self.D, use_dcn=False, with_cp=False, aspp_mid_channels=96)
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

    def get_mlp_input(self, rots, trans, intrin, post_rot, post_tran):
        B, N, _, _ = rots.shape # [B, N, 3, 3]
        mlp_input = torch.stack([
            intrin[:, :, 0, 0], # fx
            intrin[:, :, 1, 1], # fy
            intrin[:, :, 0, 2], # cx
            intrin[:, :, 1, 2], # cy
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
        ], dim=-1) # [B, N, 10]
        sensor2ego = torch.cat([rots, trans.unsqueeze(-1)], dim=-1).view(B, N, -1) # [B, N, 12]
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

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

    def get_cam_feats(self, x, mlp_input):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x) # B*N, C, h, w
        depth_x = self.depthnet(x, mlp_input) # B*N, D+C, h, w
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
        mlp_input = self.get_mlp_input(rots, trans, intrins, post_rots, post_trans)
        # B x N x D x H/downsample x W/downsample x C: cam feats
        x, depth_x, depth = self.get_cam_feats(x, mlp_input)

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

class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()