
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from model.decoder.unet3d_up import UNet3DDecoder
from segmentation_models_pytorch.base import SegmentationHead
from config.config import CFG

from model.encoder.resnet_3d import resnet_3d
from model.decoder.cnn_up import Decoder
from model.decoder.cnn_res_up import ResDecoder
from model.encoder.video_resnet_3d import r2plus1d_18, r3d_18, mc3_18
from model.decoder.uperhead import UPerHead
from model.decoder.unetplusplus_up import UnetPlusPlusDecoder
from model.encoder.convnext_3d import ConvNeXt3D
from model.encoder.convnext import ConvNeXt
from model.encoder.unet3d_down import unet_resnet_3d
from model.encoder.swin import SwinTransformer
from model.encoder.resnet_3d_v2 import resnet_3d as resnet_3d_v2

def get_model(encoder_name='resnet3d', decoder_name='cnn', inplanes=[64, 128, 256, 512], pretrained=False , **kwargs):
    if encoder_name == 'resnet3d':
        encoder = resnet_3d(18, inplanes, pretrained=pretrained, n_input_channels=1)
    elif encoder_name == 'resnet3d_v2':
        encoder = resnet_3d_v2(18, inplanes, pretrained=False, n_input_channels=1)
    elif encoder_name == 'r2plus1d_18':
        encoder = r2plus1d_18(pretrained=pretrained)
    elif encoder_name == 'r3d_18':
        encoder = r3d_18(pretrained=pretrained)
    elif encoder_name == 'mc3_18':
        encoder = mc3_18(pretrained=pretrained)
    elif encoder_name == 'convnext3d':
        encoder = ConvNeXt3D(in_chans=1, dims=inplanes)
    elif encoder_name == 'unet3d_down':
        encoder = unet_resnet_3d(18, inplanes, n_input_channels=1)
    elif encoder_name == 'swin':
        encoder = SwinTransformer(in_channels=1)
    elif encoder_name == 'convnext':
        encoder = ConvNeXt()

    if decoder_name == 'cnn':
        decoder = Decoder(inplanes, 4, kwargs['classes'])
    elif decoder_name == 'cnn_v2':
        decoder = Decoder(inplanes, 2, kwargs['classes'])
    elif decoder_name == 'uper_head':
        decoder = UPerHead(in_channels=inplanes,
                           in_index=[0, 1, 2, 3],
                           pool_scales=(1, 2, 3, 6),
                           channels=512,
                           dropout_ratio=0.1,
                           num_classes=kwargs['classes'],
                           norm_cfg=dict(type='BN', requires_grad=True),
                           align_corners=False)
    elif decoder_name == 'unetplusplus':
        decoder = UnetPlusPlusDecoder(encoder_channels=inplanes,
                                      decoder_channels=kwargs['decoder_channels'],
                                      n_blocks=kwargs['encoder_depth'],
                                      use_batchnorm=True,
                                      center=False,
                                      attention_type=kwargs['decoder_attention_type'],
                                     )
    elif decoder_name == 'unet3d_up':
        decoder = UNet3DDecoder(1, inplanes, False)
    elif decoder_name == 'cnn_res':
        decoder = ResDecoder(inplanes, 4, kwargs['classes'])
    elif decoder_name == 'cnn+uperhead':
        decoder = nn.ModuleList([UPerHead(in_channels=inplanes,
                                          in_index=[0, 1, 2, 3],
                                          pool_scales=(1, 2, 3, 6),
                                          channels=512,
                                          dropout_ratio=0.3,
                                          num_classes=kwargs['classes'],
                                          norm_cfg=dict(type='BN', requires_grad=True),
                                          align_corners=False),
                                Decoder(inplanes, 4, kwargs['classes'])])
        
    return encoder, decoder

class Ink3DModel(nn.Module):
    def __init__(self, encoder_name, decoder_name, mix_up=CFG.mix_up, **kwargs):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.mix_up = mix_up
        self.encoder, self.decoder = get_model(encoder_name=encoder_name, decoder_name=decoder_name, inplanes=CFG.inplanes, **kwargs)

    def forward(self, x):
        feat_maps = self.encoder(x)
        if self.mix_up:
            pred_mask = []
            for decoder in self.decoder:
                mask = decoder(feat_maps)
                pred_mask.append(mask)
            pred_mask = torch.stack(pred_mask, dim=0)
            pred_mask = torch.mean(pred_mask, dim=0)
        else:
            pred_mask = self.decoder(feat_maps)
        return pred_mask

class Ink3DUnet(smp.UnetPlusPlus):
    def __init__(self, encoder_name, decoder_name ,**kwargs):
        super(Ink3DUnet, self).__init__(**kwargs)
        self.encoder_name = encoder_name
        self.encoder, self.decoder = get_model(encoder_name=encoder_name, decoder_name=decoder_name, **kwargs)
        self.segmentation_head = SegmentationHead(
            in_channels=kwargs['decoder_channels'][-1],
            out_channels=kwargs['classes'],
            activation=None,
            kernel_size=3,
            upsampling=2
        )
