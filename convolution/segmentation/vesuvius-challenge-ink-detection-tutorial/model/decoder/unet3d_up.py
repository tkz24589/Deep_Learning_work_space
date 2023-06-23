import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.identity_map = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        residual = self.identity_map(x)
        out = out + residual
        return out
    
class Up3d(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up3d, self).__init__()

        if bilinear:
            # 使用双线性上采样进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            # 使用反卷积进行上采样
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    
        self.conv = nn.Sequential(DoubleConv3d(in_channels, out_channels),
                                  nn.BatchNorm3d(out_channels),
                                  nn.GELU())

    def forward(self, x1, x2):
        # 进行上采样
        x1 = self.up(x1)
        # 在深度、高度、宽度的维度上进行padding，使得上采样后的形状与下采样时相同
        diffZ = x2.shape[2] - x1.shape[2]
        diffY = x2.shape[3] - x1.shape[3]
        diffX = x2.shape[4] - x1.shape[4]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2))
        # 将下采样和上采样的特征图进行拼接，并进行卷积操作
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UNet3DDecoder(nn.Module):
    def __init__(self, n_classes, inplanes , bilinear=True):
        super(UNet3DDecoder, self).__init__()
        self.bilinear = bilinear
        self.up1 = Up3d(inplanes[3], inplanes[2], bilinear)
        self.up2 = Up3d(inplanes[2], inplanes[1], bilinear)
        self.up3 = Up3d(inplanes[1], inplanes[0], bilinear)
        self.outc = nn.Conv3d(64, n_classes, kernel_size=1)
        # self.upsample = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, x):
        x1, x2, x3, x4 = x
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        logits = F.interpolate(logits, scale_factor=4, mode='trilinear')
        logits = torch.mean(logits, dim=2)
        return logits