# import torch.nn as nn
# import torch

# # class ConvNeXtBlock3D(nn.Module):
# #     def __init__(self, in_channels, out_channels, cardinality, bottleneck_width, stride, downsample=False):
# #         super(ConvNeXtBlock3D, self).__init__()

# #         self.downsample = downsample
# #         self.stride = stride

# #         bottleneck_channels = out_channels * bottleneck_width // 32      # 计算瓶颈层的输出通道数

# #         # Cardinality group convolution
# #         self.conv1 = nn.Conv3d(in_channels, bottleneck_channels * cardinality, kernel_size=1, stride=1, bias=False)
# #         self.bn1 = nn.BatchNorm3d(bottleneck_channels * cardinality)
# #         self.relu1 = nn.ReLU(inplace=True)

# #         self.conv2 = nn.Conv3d(bottleneck_channels * cardinality, bottleneck_channels * cardinality, 
# #                                kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
# #         self.bn2 = nn.BatchNorm3d(bottleneck_channels * cardinality)
# #         self.relu2 = nn.ReLU(inplace=True)

# #         self.conv3 = nn.Conv3d(bottleneck_channels * cardinality, out_channels, kernel_size=1, stride=1, bias=False)
# #         self.bn3 = nn.BatchNorm3d(out_channels * bottleneck_width)

# #         # Downsample
# #         if self.downsample:
# #             self.downsample_conv = nn.Conv3d(in_channels, out_channels * bottleneck_width, kernel_size=1, stride=stride, bias=False)
# #             self.downsample_bn = nn.BatchNorm3d(out_channels * bottleneck_width)

# #     def forward(self, x):
# #         residual = x

# #         out = self.conv1(x)
# #         out = self.bn1(out)
# #         out = self.relu1(out)

# #         out = self.conv2(out)
# #         out = self.bn2(out)
# #         out = self.relu2(out)

# #         out = self.conv3(out)
# #         out = self.bn3(out)

# #         if self.downsample:
# #             residual = self.downsample_conv(x)
# #             residual = self.downsample_bn(residual)

# #         out += residual
# #         out = nn.ReLU(inplace=True)(out)
# #         return out

# # class ConvNeXt3D(nn.Module):
# #     def __init__(self, in_channels, layers, block_inplanes, cardinality=32, bottleneck_width=1, **kwargs):
# #         super(ConvNeXt3D, self).__init__()

# #         self.cardinality = cardinality
# #         self.bottleneck_width = bottleneck_width

# #         self.in_planes = block_inplanes[0]
# #         self.output_stride = 16
# #         # Stem layer
# #         self.stem = nn.Sequential(
# #             nn.Conv3d(in_channels, self.in_planes, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3)),
# #             nn.BatchNorm3d(self.in_planes),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
# #         )

# #         # ConvNeXt blocks
# #         self.stage1 = self._make_stage(block_inplanes[0], block_inplanes[0], num_blocks=layers[0], stride=(1, 1, 1))
# #         self.stage2 = self._make_stage(block_inplanes[0], block_inplanes[1], num_blocks=layers[1], stride=(1, 2, 2))
# #         self.stage3 = self._make_stage(block_inplanes[1], block_inplanes[2], num_blocks=layers[2], stride=(1, 2, 2))
# #         self.stage4 = self._make_stage(block_inplanes[2], block_inplanes[3], num_blocks=layers[3], stride=(1, 2, 2))


# #     def _make_stage(self, in_channels, out_channels, num_blocks, stride):
# #         layers = []
# #         layers.append(ConvNeXtBlock3D(in_channels, out_channels, self.cardinality, self.bottleneck_width, 
# #                                       stride=stride, downsample=True))
# #         for i in range(1, num_blocks):
# #             layers.append(ConvNeXtBlock3D(out_channels * self.bottleneck_width, out_channels, self.cardinality, 
# #                                           self.bottleneck_width, stride=1, downsample=False))
# #         return nn.Sequential(*layers)

# #     def forward(self, x):
# #         x = self.stem(x)
# #         x1 = self.stage1(x)
# #         x2 = self.stage2(x1)
# #         x3 = self.stage3(x2)
# #         x4 = self.stage4(x3)
# #         out = [x1, x2 ,x3, x4]
# #         out = [torch.mean(i, dim=2) for i in out]
# #         return out

# import torch
# import torch.nn as nn


# class DenseBlock3d(nn.Module):
#     def __init__(self, in_channels, growth_rate, layer_depth):
#         super(DenseBlock3d, self).__init__()
#         self.layer_depth = layer_depth
#         self.conv_layers = nn.ModuleList()

#         for i in range(self.layer_depth):
#             in_channels += i * growth_rate
#             self.conv_layers.append(
#                 nn.Conv3d(
#                     in_channels,
#                     growth_rate,
#                     kernel_size=(3, 3, 3),
#                     padding=(1, 1, 1),
#                 )
#             )
#         self.out_channel = in_channels + growth_rate
#     def forward(self, x):
#         out = x
#         for layer in self.conv_layers:
#             out = torch.cat([out, layer(out)], dim=1)
#         return out


# class ConvNext3d(nn.Module):
#     def __init__(
#         self,
#         in_channels=1,
#         growth_rate=64,
#         num_layers=[2, 2, 2, 2]
#     ):
#         super(ConvNext3d, self).__init__()
#         self.conv1 = nn.Conv3d(
#             in_channels,
#             64,
#             kernel_size=(3, 3, 3),
#             stride=(1, 2, 2),
#             padding=(1, 1, 1),
#         )
#         # self.maxpool1 = nn.MaxPool3d(
#         #     kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1)
#         # )

#         self.dense_blocks = nn.ModuleList()
#         self.transition_layers = nn.ModuleList()

#         for i in range(len(num_layers)):
#             if i == 0:
#                 in_channels = 64
#             else:
#                 in_channels = in_channels + 2 * growth_rate

#             dense_block = nn.Sequential(
#                 DenseBlock3d(
#                     in_channels=in_channels,
#                     growth_rate=growth_rate,
#                     layer_depth=num_layers[i]),
#                 nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
#             )
#             self.dense_blocks.append(dense_block)

#             transition_layer = nn.Sequential(
#                 nn.Conv3d(
#                     dense_block[0].out_channel,
#                     2**i * growth_rate,
#                     kernel_size=(1, 1, 1),
#                     stride=(1, 1, 1),
#                 ),
#                 nn.BatchNorm3d(2**i * growth_rate),
#                 nn.GELU()
#             )
#             self.transition_layers.append(transition_layer)


#     def forward(self, x):
#         x = self.conv1(x)
#         # x = self.maxpool1(x)
#         x1 = self.dense_blocks[0](x)
#         x1_out = self.transition_layers[0](x1)

#         x2 = self.dense_blocks[1](x1)
#         x2_out = self.transition_layers[1](x2)

#         x3 = self.dense_blocks[2](x2)
#         x3_out = self.transition_layers[2](x3)

#         x4 = self.dense_blocks[3](x3)
#         x4_out = self.transition_layers[3](x4)

#         out = [x1_out, x2_out, x3_out, x4_out]
#         out = [torch.mean(i, dim=2) for i in out]

#         return out
    
#     def get_pooling_dim(self, dim_size):
#         return int(dim_size / 2)

# def convnext_3d(n_input_channels=1):
#     model = ConvNext3d(in_channels=n_input_channels)
#     return model


# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmseg".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """

    logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)

    return logger

class BayarConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv3d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    # def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
    #     super().__init__()
    #     self.dwconv = nn.Conv3d(dim, dim, kernel_size=(7, 7, 7) ,padding=(3, 3, 3), groups=dim, bias=False) # depthwise conv
    #     self.norm = LayerNorm(dim, eps=1e-6)
    #     self.pwconv1 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False) # pointwise/1x1 convs, implemented with linear layers
    #     self.act = nn.GELU()
    #     self.pwconv2 = nn.Conv3d(dim, dim, kernel_size=1, stride=1 ,  groups=dim, bias=False) 
    #     # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
    #     #                             requires_grad=True) if layer_scale_init_value > 0 else None
    #     self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, D, H, W, C) -> (N, C, D, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class ConvNeXt3D(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, bayar=False, depths=[2, 2, 2, 2], dims=[64, 128, 256, 512], 
                 drop_path_rate=0.1, layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], pretrained=None
                 ):
        super().__init__()
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.bayar = bayar
        if self.bayar:
            self.bayar_conv = BayarConv3d(in_chans, in_chans)
            self.in_chans = in_chans * 2

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(self.in_chans, dims[0], kernel_size=(7, 7, 7), stride=(1, 4, 4), padding=(3, 3, 3), bias=False),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=(3,3,3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is None:
            pretrained = self.pretrained

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        if self.bayar:
            x_bayar = self.bayar_conv(x)
            x = torch.cat([x, x_bayar], dim=1)
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        outs = [torch.mean(i, dim=2) for i in outs]

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x
    
if __name__ == '__main__':
    i = torch.ones(8, 1, 20, 224, 224).cuda().float()
    model = ConvNeXt3D(in_chans=1).cuda()
    print(model)
    o = model(i)
    print()