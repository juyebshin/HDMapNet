from cv2 import norm
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18, resnet50


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class UpDT(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x1, x2):
        # x1: feature
        # x2: distance transform
        x1 = self.up(x1)
        x1 = torch.cat([x1, x2], dim=1)
        return self.conv(x1) # b, 3, 200, 400


class CamEncode(nn.Module):
    def __init__(self, C, backbone='efficientnet-b4', norm_layer=nn.BatchNorm2d):
        super(CamEncode, self).__init__()
        self.C = C
        self.backbone = backbone

        if 'efficientnet' in backbone:
            self.trunk = EfficientNet.from_pretrained(backbone)
        elif backbone == 'resnet-18':
            self.trunk = resnet18(pretrained=True)
        elif backbone == 'resnet-50':
            self.trunk = resnet50(pretrained=True)
        else:
            raise NotImplementedError
        
        if backbone == 'efficientnet-b0':
            channel = 320+112
        elif backbone == 'efficientnet-b4':
            channel = 448+160
        elif backbone == 'efficientnet-b7':
            channel = 640+224
        elif backbone == 'resnet-18':
            channel = 512+256
        elif backbone == 'resnet-50':
            channel = 2048+1024
        else:
            raise NotImplementedError

        self.up1 = Up(channel, self.C, norm_layer=norm_layer) # 320+112

        """
        b0
        reduction_1: torch.Size([1, 16, 112, 112])
        reduction_2: torch.Size([1, 24, 56, 56])
        reduction_3: torch.Size([1, 40, 28, 28])
        reduction_4: torch.Size([1, 112, 14, 14])
        reduction_5: torch.Size([1, 320, 7, 7])
        reduction_6: torch.Size([1, 1280, 7, 7])

        b4
        reduction_1: torch.Size([1, 24, 112, 112])
        reduction_2: torch.Size([1, 32, 56, 56])
        reduction_3: torch.Size([1, 56, 28, 28])
        reduction_4: torch.Size([1, 160, 14, 14])
        reduction_5: torch.Size([1, 448, 7, 7])
        reduction_6: torch.Size([1, 1792, 7, 7])

        b7
        reduction_1: torch.Size([1, 32, 112, 112])
        reduction_2: torch.Size([1, 48, 56, 56])
        reduction_3: torch.Size([1, 80, 28, 28])
        reduction_4: torch.Size([1, 224, 14, 14])
        reduction_5: torch.Size([1, 640, 7, 7])
        reduction_6: torch.Size([1, 2560, 7, 7])

        r18
        x1: torch.Size([1, 64, 56, 56])
        x2: torch.Size([1, 128, 28, 28])
        x3: torch.Size([1, 256, 14, 14])
        x4: torch.Size([1, 512, 7, 7])
        
        r50
        x1: torch.Size([1, 256, 56, 56])
        x2: torch.Size([1, 512, 28, 28])
        x3: torch.Size([1, 1024, 14, 14])
        x4: torch.Size([1, 2048, 7, 7])
        """

    def get_eff_depth(self, x):
        # x: B*N, C, H, W
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        # Conv -> BN -> Swish
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def get_resnet_depth(self, x):
        # x: B*N, C, H, W
        # adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L266
        x = self.trunk.conv1(x) # [B*N, 64, H/2, W/2]
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x) # [B*N, 64, H/4, W/4]

        x1 = self.trunk.layer1(x) # [B*N, 64 or 246, H/4, W/4]
        x2 = self.trunk.layer2(x1) # [B*N, 128 or 512, H/8, W/8]
        x3 = self.trunk.layer3(x2) # [B*N, 256 or 1024, H/16, W/16]
        x4 = self.trunk.layer4(x3) # [B*N, 512 or 2048, H/32, W/32]

        x = self.up1(x4, x3)
        return x

    def forward(self, x):
        if 'efficientnet' in self.backbone:
            return self.get_eff_depth(x)
        elif 'resnet' in self.backbone:
            return self.get_resnet_depth(x)
        else:
            raise NotImplementedError


class BevEncode(nn.Module):
    def __init__(self, inC, outC, norm_layer=nn.BatchNorm2d, segmentation=True, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=37, distance_reg=True, vertex_pred=True):
        super(BevEncode, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4, norm_layer=norm_layer)

        self.segmentation = segmentation
        self.up2 = nn.Sequential( # final semantic segmentation prediction
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0), # outC = 4 (num_classes)
        )

        self.distance_reg = distance_reg
        if distance_reg:
            # self.up1_dt = Up(64 + 256, 256, scale_factor=4)
            self.up_dt = nn.Sequential( # distance transform prediction
                # b, 256, 100, 200
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                # b, 256, 200, 400
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                # b, 128, 200, 400
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, outC-1, kernel_size=1, padding=0), # outC = 3 no background
                # b, 3, 200, 400
            )
            self.up3 = UpDT(256 + outC-1, outC, scale_factor=2, norm_layer=norm_layer)
        else:
            self.up_bin = nn.Sequential( # distance transform prediction
                # b, 256, 100, 200
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                # b, 256, 200, 400
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                # b, 128, 200, 400
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, outC-1, kernel_size=1, padding=0), # outC = 3 no background
                # b, 3, 200, 400
            )

        self.vertex_pred = vertex_pred
        if vertex_pred:
            self.vertex_head = nn.Sequential(
                # b, 256, 100, 200
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                # b, 128, 100, 200
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # b, 256, 50, 100
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False), # 65: cell_size*cell_size + 1 (dustbin)
                # b, 128, 50, 100
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # b, 128, 25, 50
                nn.Conv2d(128, 65, kernel_size=1, padding=0), # 65: cell_size*cell_size + 1 (dustbin)
                # b, 65, 25, 50
            )

        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(64 + 256, 256, scale_factor=4, norm_layer=norm_layer)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

        self.direction_pred = direction_pred
        if direction_pred:
            self.up1_direction = Up(64 + 256, 256, scale_factor=4, norm_layer=norm_layer)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1, padding=0),
            )

    def forward(self, x): # x: b, 64, 200, 400
        x = self.conv1(x) # b, 64, 100, 200
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) # b, 64, 100, 200
        x = self.layer2(x1) # b, 128, 50, 100
        x2 = self.layer3(x) # b, 256, 25, 50

        x = self.up1(x2, x1) # b, 256, 100, 200, apply distance transform after here

        if self.vertex_pred:
            x_vertex = self.vertex_head(x) # b, 65, 25, 50
        else:
            x_vertex = None

        if self.distance_reg:
            x_dt = self.up_dt(x) # b, 3, 200, 400
            # x: [b, 256, 100, 200], x_dt: [b, 3, 200, 400]
            # concat [x, x_dt] and upsample to get dense semantic prediction
            if self.segmentation:
                x_seg = self.up3(x, self.relu(x_dt)) # b, 4, 200, 400
            else:
                x_seg = None
        else:
            x_dt = None # b, 4, 200, 400 # semantic segmentation prediction
            if self.segmentation:
                x_seg = self.up2(x) # b, 4, 200, 400 # semantic segmentation prediction
            else:
                x_seg = None
        # x = self.up2(x) # b, 4, 200, 400 # semantic segmentation prediction
        
        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1) # b, 256, 100, 200
            x_embedded = self.up2_embedded(x_embedded) # b, 16, 200, 400
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_embedded(x2, x1)
            x_direction = self.up2_direction(x_direction) # b, 37, 200, 400
        else:
            x_direction = None

        return x_seg, x_dt, x_vertex, x_embedded, x_direction
