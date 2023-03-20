# # encoding:utf-8
# # Modify from torchvision
# # ResNeXt: Copy from https://github.com/last-one/tools/blob/master/pytorch/SE-ResNeXt/SeResNeXt.py
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import init
#
# import config
# from models.feature_net import get_resnet50, get_senet
# from models.smpl import SMPL
# from models.transformer import Transformer
# from utils.geometry import rot6d_to_rotmat
#
# """
# This file contains the definitions of the various ResNet models.
# Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
# Forward pass was modified to discard the last fully connected layer
# """
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         # remove final fully connected layer
#         # x = self.fc(x)
#
#         return x
#
#
# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model
#
#
# class HPR(nn.Module):
#     def __init__(self, smpl_mean_params=None):
#         super(HPR, self).__init__()
#         self.cam_size = 3
#         self.pose_size = 144
#         self.shape_size = 10
#         self.smpl = SMPL(config.SMPL_MODEL_DIR,
#                          create_transl=False)
#         self.feature_net = get_senet()
#         self._init_params()
#         self.pose_tranformer = Transformer(dim=2048 + self.pose_size, depth=3, heads=2,
#                                            dim_head=8, mlp_dim=3072, out_dim=self.pose_size)
#
#         self.shape_tranformer = Transformer(dim=2048 + self.shape_size, depth=3, heads=2,
#                                             dim_head=8, mlp_dim=3072, out_dim=self.shape_size)
#
#         self.cam_tranformer = Transformer(dim=2048 + self.cam_size, depth=3, heads=2,
#                                           dim_head=8, mlp_dim=3072, out_dim=self.cam_size)
#
#         mean_params = np.load(smpl_mean_params)
#         init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
#         init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
#         init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
#         self.register_buffer('init_pose', init_pose)
#         self.register_buffer('init_shape', init_shape)
#         self.register_buffer('init_cam', init_cam)
#
#     def _init_params(self):
#         for name, module in self.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 init.kaiming_uniform_(module.weight)
#                 if module.bias is not None:
#                     init.constant_(module.bias, 0)
#
#     def forward(self, x, init_pose=None, init_shape=None, init_cam=None):
#         """
#         :param image:
#         :return:
#         """
#         batch_size = x.shape[0]
#
#         if init_pose is None:
#             init_pose = self.init_pose.expand(batch_size, 1, -1)
#         if init_shape is None:
#             init_shape = self.init_shape.expand(batch_size, 1, -1)
#         if init_cam is None:
#             init_cam = self.init_cam.expand(batch_size, 1, -1)
#
#         feature = self.feature_net(x)
#         pred_pose = self.pose_tranformer(torch.cat([feature, init_pose], 2)).squeeze(1)
#         pred_shape = self.shape_tranformer(torch.cat([feature, init_shape], 2)).squeeze(1)
#         pred_cam = self.cam_tranformer(torch.cat([feature, init_cam], 2)).squeeze(1)
#         pred_pose = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
#         return pred_pose, pred_shape, pred_cam
#
#
# def hpr(smpl_mean_params=None,
#         pretrained=False,
#         checkpoint_file=None):
#     """ Constructs an HMR model with ResNet50 backbone.
#         Args:
#             pretrained (bool): If True, returns a model pre-trained on ImageNet
#             :param checkpoint_file:
#             :param smpl_mean_params:
#         """
#     model = HPR(smpl_mean_params)
#     if pretrained:
#         checkpoint = torch.load(checkpoint_file)
#         model.load_state_dict(checkpoint['model'], strict=False)
#     return model
# encoding:utf-8
# Modify from torchvision
# ResNeXt: Copy from https://github.com/last-one/tools/blob/master/pytorch/SE-ResNeXt/SeResNeXt.py
import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.nn import init

import config
import constants
from models.feature_net import get_resnet50, get_senet
from models.smpl import SMPL
from models.transformer import Transformer
from models.uper_head import UPerHead
from utils.geometry import rot6d_to_rotmat
from utils.pointnetEncoder import PointNet, get_image_pointset

"""
This file contains the definitions of the various ResNet models.
Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
Forward pass was modified to discard the last fully connected layer
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        # remove final fully connected layer
        # x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class HPR(nn.Module):
    def __init__(self, smpl_mean_params=None, type='all', point_num=[256, 512, 1024, 2048]):
        super(HPR, self).__init__()
        self.cam_size = 3
        self.pose_size = 144
        self.shape_size = 10
        self.type = type
        self.npose_per_joint = 6
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         create_transl=False)
        self.feature_net = resnet50(pretrained=True)

        self.point_net_1 = PointNet(point_num=point_num[0])
        self.point_net_2 = PointNet(point_num=point_num[1])
        self.point_net_3 = PointNet(point_num=point_num[2])
        self.point_net_4 = PointNet(point_num=point_num[3])

        self.point_docoder = UPerHead(in_channels=[point_num[0], point_num[1], point_num[2], point_num[3]],
                                    in_index=[0, 1, 2, 3],
                                    pool_scales=(1, 2, 3, 6),
                                    channels=1088,
                                    dropout_ratio=0.2,
                                    num_classes=1,
                                    norm_cfg=dict(type='BN', requires_grad=True),
                                    align_corners=False)

        self.pose_attention = Attention(dim=3136, heads=8, dim_head=64)

        self.shape_attention = Attention(dim=2048, heads=8, dim_head=64)

        self.cam_attention = Attention(dim=2048, heads=8, dim_head=64)

        self.pose_regressor = nn.Sequential(
            nn.Linear(1088 + 2048 + self.pose_size, 2048),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.Dropout(),
            # nn.Linear(1024, self.pose_size)
        )
        self.joint_regs = nn.ModuleList()
        for joint_idx, ancestor_idx in enumerate(constants.ANCESTOR_INDEX):
            regressor = nn.Linear(1024 + self.npose_per_joint * len(ancestor_idx), self.npose_per_joint)
            nn.init.xavier_uniform_(regressor.weight, gain=0.01)
            self.joint_regs.append(regressor)

        self.shape_regressor = nn.Sequential(
            nn.Linear(2048 + self.shape_size, 1024),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.Linear(1024, self.shape_size)
        )
        self.cam_regressor = nn.Sequential(
            nn.Linear(2048 + self.cam_size, 1024),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.Linear(1024, self.cam_size)
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, point, init_pose=None, init_shape=None, init_cam=None):
        """
        :param image:
        :return:
        """
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        feature = self.feature_net(x)
        point_feature_1 = self.point_net_1(point[0]).transpose(2, 1).unsqueeze(3)
        point_feature_2 = self.point_net_2(point[1]).transpose(2, 1).unsqueeze(3)
        point_feature_3 = self.point_net_3(point[2]).transpose(2, 1).unsqueeze(3)
        point_feature_4 = self.point_net_4(point[3]).transpose(2, 1).unsqueeze(3)

        point_feature = self.point_docoder([point_feature_1, point_feature_2, point_feature_3, point_feature_4]).squeeze()

        global_feature = torch.cat([feature, point_feature], dim=1).unsqueeze(1)
        
        feature = feature.unsqueeze(1)

        x1 = global_feature + self.pose_attention(global_feature)
        x2 = feature + self.shape_attention(feature)
        x3 = feature + self.cam_attention(feature)

        x1 = self.pose_regressor(torch.cat([x1.squeeze(1), init_pose], dim=1))
        pose = []
        for ancestor_idx, reg in zip(constants.ANCESTOR_INDEX, self.joint_regs):
            ances = torch.cat([x1] + [pose[i] for i in ancestor_idx], dim=1)
            pose.append(reg(ances))
        pred_pose = torch.cat(pose, dim=1)

        if self.type == 'pose':
            pred_shape = init_shape
            pred_cam = init_cam
        else:
            pred_shape = self.shape_regressor(torch.cat([x2.squeeze(1), init_shape], dim=1))
            pred_cam = self.cam_regressor(torch.cat([x3.squeeze(1), init_cam], dim=1))

        pred_pose = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        return pred_pose, pred_shape, pred_cam


def hpr_pointnet_net(smpl_mean_params=None,
        pretrained=False,
        checkpoint_file=None,
        type='all',
        point_num=64):
    """ Constructs an HMR model with ResNet50 backbone.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            :param checkpoint_file:
            :param smpl_mean_params:
        """
    if type != 'all':
        type = 'pose'
    model = HPR(smpl_mean_params=smpl_mean_params, type=type, point_num=point_num)
    if pretrained:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model

if __name__== "__main__" :
    net = hpr_pointnet_net(smpl_mean_params=config.SMPL_MEAN_PARAMS)
    images = cv2.imread('datasets/data/sketch-mpi-inf-3dhp/S1/Seq2/imageFrames/video_0/frame_000002_canny.png')[:, :,
    ::-1].astype(np.float32)
    images = cv2.resize(images, (224, 224))
    cv2.imshow('a', images)
    point = get_image_pointset(torch.from_numpy(images).float(), npoint=64).unsqueeze(0).float()
    img = np.zeros((224, 224))
    for i in point[0]:
        img[int(i[0])][int(i[1])] = 255
    # cv2.imshow('b',img)
    # cv2.waitKey(0)
    images = torch.from_numpy(images).unsqueeze(0).reshape(-1, 3, 224, 224)
    net(images, point)