import cv2
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from model_seg.decode_heads.uper_head import UPerHead
from model_seg.backbones.swin import SwinTransformer

from utils.utils import dice_coef, iou_coef
from tqdm import tqdm
from utils.dataset import DTTDataSet
from unet.unet_model import Res_Unet
from torch.utils.data import DataLoader
from model_seg.ops import resize
from model_seg.losses.accuracy import Accuracy
from torchvision.utils import make_grid


def eval_net(net, dataset, size, stride, gpu=True, score_min=0.5, is_slice=False, slice=256, net_name="class1",
             model="", writer=None, summerry_size=30):
    """Evaluation without the densecrf with the dice coefficient"""
    iou_tot = 0
    dice_tot = 0
    score = 0
    net.cuda()
    accuracy = Accuracy(topk=1)
    for i, (img, true_mask, flag) in tqdm(enumerate(dataset), total=len(dataset)):
        if is_slice:
            img = img.view(-1, 3, slice, slice)
            true_mask = true_mask.view(-1, 1, slice, slice)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        with torch.no_grad():
            mask_pred = net(img)

            mask_pred = resize(input=mask_pred,
                               size=true_mask.shape[2:],
                               mode='bilinear',
                               align_corners=False)
            sc, _, _ = accuracy(mask_pred, true_mask.squeeze(1))
            score += sc.item()
            pred_normal = torch.sigmoid(mask_pred[:, :1, :]).float()
            pred_forge = torch.sigmoid(mask_pred[:, 1:, :]).float()
            if i % summerry_size == 0:
                # 使用make_grid将图片转换成网格形式
                pred_forge = make_grid(pred_forge, global_step=i, normalize=True)
                pred_normal = make_grid(pred_normal, global_step=i, normalize=True)
                mask = make_grid(true_mask.float(), global_step=i, normalize=True)
                # 使用add_image方法将图片添加到TensorBoard中
                writer.add_image('Valid/forge', pred_forge, dataformats="CHW")
                writer.add_image('Valid/normal', pred_normal, dataformats="CHW")
                writer.add_image('Valid/mask', mask, dataformats="CHW")

            iou_tot += iou_coef(y_true=true_mask, y_pred=torch.sigmoid(mask_pred[:, 1:, :]),
                                thr=score_min).item()
            dice_tot += dice_coef(y_true=true_mask, y_pred=torch.sigmoid(mask_pred[:, 1:, :]),
                                  thr=score_min).item()
    iou_score = iou_tot / len(dataset)
    dice_score = dice_tot / len(dataset)
    scores = score / len(dataset)
    return iou_score, dice_score, scores


if __name__ == "__main__":
    n_class = 2
    batchsize = 6
    epochs = 50
    size = 256
    crop_size = 256
    classes = ["0", "1"]
    palette = [[0, 0, 0], [255, 255, 255]]
    valid_dataset = DTTDataSet(is_train=False,
                               data_root='/home/stu/Deep_Learning_work_space/convolution/segmentation/dtt/data/val/tampered',
                               img_dir='imgs',
                               ann_dir='masks',
                               classes=classes,
                               palette=palette,
                               size=size,
                               crop_size=crop_size
                               )
    net = nn.Sequential(SwinTransformer(pretrain_img_size=224,
                                        embed_dims=128,
                                        depths=[2, 2, 9, 2],
                                        num_heads=[4, 8, 16, 16],
                                        window_size=7,
                                        use_abs_pos_embed=False,
                                        drop_path_rate=0.3,
                                        patch_norm=True,
                                        patch_size=4,
                                        mlp_ratio=4,
                                        strides=(4, 2, 2, 2),
                                        out_indices=(0, 1, 2, 3),
                                        qkv_bias=True,
                                        qk_scale=None,
                                        drop_rate=0.1,
                                        attn_drop_rate=0.1,
                                        act_cfg=dict(type='GELU'),
                                        norm_cfg=dict(type='LN', requires_grad=True)),
                        UPerHead(in_channels=[128, 256, 512, 1024],
                                 in_index=[0, 1, 2, 3],
                                 pool_scales=(1, 2, 3, 6),
                                 channels=512,
                                 dropout_ratio=0.1,
                                 num_classes=2,
                                 norm_cfg=dict(type='BN', requires_grad=True),
                                 align_corners=False),
                        # FCNHead(in_channels=512,
                        #         in_index=2,
                        #         channels=256,
                        #         num_convs=1,
                        #         concat_input=False,
                        #         dropout_ratio=0.1,
                        #         num_classes=n_calss,
                        #         norm_cfg=dict(type='BN', requires_grad=True),
                        #         align_corners=False)
                        ).cuda()

    # fine_tuning_model = 'result/DDT-[iou_score]-0.0969-[dice_score]-0.1726-[train_loss]-0.7742.pkl'
    # net.load_state_dict(torch.load(fine_tuning_model))
    # print('Model loaded from {}'.format(fine_tuning_model))

    valid_loader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    iou_score, dice_score, score = eval_net(net, valid_loader, 512, 128, True, 0.5, False, slice, model="Swin")
    print(score)
