import gc
import json
import torch
import torch.backends.cudnn as cudnn
from torch import optim

from eval import eval_net
from model_seg.backbones.convnext import ConvNeXt
from unet.unet_model import Ringed_Res_Unet, Unet
from model.res_unet import ResUnet
from model.unet_3plus import UNet_3Plus
from model.mvss_net import get_mvss
from model.unet_plus_plus import HFUnetPlusPlus
from utils import *
from torch.optim import lr_scheduler
from transformers import PretrainedConfig
from model.fcn16s import FCN16s
from model_seg.backbones.swin import SwinTransformer
from model_seg.decode_heads.uper_head import UPerHead
from model_seg.decode_heads.fcn_head import FCNHead
from model_seg.samper.ohem_pixel_sampler import OHEMPixelSampler
from model_seg.ops import resize
from model_seg.losses.lovasz_loss import LovaszLoss
from model_seg.losses.cross_entropy_loss import CrossEntropyLoss
from model_seg.losses.dice_loss import DiceLoss

import matplotlib.pyplot as plt
import time
import cv2
from tqdm import tqdm

import segmentation_models_pytorch as smp
from torch.cuda import amp
from utils.dataset import DTTDataSet
from torch.utils.data import DataLoader
import warnings
from colorama import Fore, Style
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

warnings.filterwarnings("ignore")
c_ = Fore.GREEN
sr_ = Style.RESET_ALL


def train_net(net,
              train_dataset,
              valid_dataset,
              model,
              epochs=1,
              batch_size=1,
              lr=0.1,
              save_cp=True,
              gpu=False,
              slice=256,
              stride=128,
              is_slice=False,
              size=384,
              n_calss=1,
              dir_logs='result/',
              summery_size=150):
    # training images are square

    print('''
    Starting training:
        Model: {}
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
        Img size: {}
    '''.format(model,
               epochs,
               batch_size,
               lr,
               len(train_dataset),
               len(valid_dataset),
               str(save_cp),
               str(gpu),
               str(size)))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

    # 实例化SummaryWriter对象
    writer = SummaryWriter('result/logs')

    optimizer = optim.AdamW(net.parameters(),
                            lr=lr,
                            betas=(0.9, 0.999),
                            weight_decay=0.05
                            )
    T_max = 500
    min_lr = 0.000001
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)
    scaler = amp.GradScaler()

    sampler = OHEMPixelSampler(ignore_index=1, thresh=0.9, min_kept=1000)

    '''Loss Function'''
    # cross_entropy_loss = nn.CrossEntropyLoss()
    # lovasz_loss = smp.losses.LovaszLoss(mode='multiclass', per_image=True)
    # dice_loss = smp.losses.DiceLoss(mode='multiclass')
    cross_entropy_loss = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)
    lovasz_loss = LovaszLoss(per_image=True, loss_weight=1.0)
    dice_loss = DiceLoss(class_weight=[0.1, 1.0], loss_weight=0.4)

    Train_loss = []
    Iou_score = []
    Dice_score = []
    scores = []
    EPOCH = []
    best_score = 0
    accumulation_steps = 1

    for epoch in tqdm(range(epochs), total=epochs):
        net.train()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4)

        epoch_loss = 0

        bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (imgs, masks_true, flag) in bar:

            if gpu:
                imgs = imgs.cuda()
                masks_true = masks_true.cuda()

            with amp.autocast():

                masks_pred = net(imgs)

                masks_pred = resize(input=masks_pred,
                                    size=masks_true.shape[2:],
                                    mode='bilinear',
                                    align_corners=False)
                if sampler is not None:
                    seg_weight = sampler.sample(masks_pred, masks_true)

                # pred = torch.sigmoid(masks_pred[0][1]).float()
                # cv2.imwrite('train_pred.png', pred.detach().cpu().numpy() * 255)
                # cv2.imwrite('train_true.png', masks_true[0][0].detach().cpu().numpy())
                if i % summery_size == 0:
                    # 使用make_grid将图片转换成网格形式
                    grid_image = make_grid(torch.sigmoid(masks_pred[:, 1:, :, :]), normalize=True)
                    grid_mask = make_grid(masks_true.float(), normalize=True)
                    # 使用add_image方法将图片添加到TensorBoard中
                    writer.add_image('Train/Image', grid_image, global_step=i, dataformats="CHW")
                    writer.add_image('Train/mask', grid_mask, global_step=i, dataformats="CHW")
                masks_true = masks_true.squeeze(1)

                loss = (cross_entropy_loss(masks_pred, masks_true, weight=seg_weight)
                        + lovasz_loss(masks_pred, masks_true)) / accumulation_steps

            scaler.scale(loss).backward(retain_graph=True)

            # 更新梯度
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            bar.set_postfix(loss=f'{loss.item():0.2e}', epoch=epoch, gpu_mem=f'{mem:0.2f} GB',
                            lr=f'{optimizer.state_dict()["param_groups"][0]["lr"]:0.2e}')
            epoch_loss += loss.item()
        print()
        print('Epoch finished ! Loss: {:.4f}'.format(epoch_loss / len(train_loader)))

        # validate the performance of the model
        gc.collect()
        torch.cuda.empty_cache()

        net.eval()

        iou_score, dice_score, score = eval_net(net, valid_loader, size, stride, gpu, 0.5, is_slice, slice, model=model,
                                                writer=writer)
        print()
        print('Iou Score: {:.4f}'.format(iou_score))
        print('Dice Score: {:.4f}'.format(dice_score))
        print('Score: {:.4f}'.format(score))

        writer.add_scalar('Train/Loss', epoch_loss / len(train_loader), epoch)
        writer.add_scalar('Test/IOU', iou_score, epoch)
        writer.add_scalar('Test/Dice', dice_score, epoch)
        writer.add_scalar('Test/Score', score, epoch)

        if iou_score > best_score:
            print(f"{c_}Valid Score Improved ({best_score:0.4f} ---> {iou_score:0.4f})")
            torch.save(net.state_dict(),
                       dir_logs + '{}-[iou_score]-{:.4f}-[train_loss]-{:.4f}.pkl'.format(model_name, iou_score,
                                                                                         epoch_loss / len(
                                                                                             train_loader)))
            best_score = iou_score
            print(f"Model Saved{sr_}")
        if (epoch + 1) % 10 == 0:
            print(f"{c_}Valid Score {iou_score:0.4f}")
            torch.save(net.state_dict(),
                       dir_logs + '{}-[iou_score]-{:.4f}-[train_loss]-{:.4f}-'.format(model_name, iou_score,
                                                                                      epoch_loss / len(
                                                                                          train_loader)) + str(
                           epoch) + '_epoch.pkl')
            print(f"Last Model Saved{sr_}")
    writer.close()


if __name__ == '__main__':
    # n_calss, epochs, batchsize, size, slice, stride, is_slice, gpu = 2, 100, 2, 512, 512, 512, False, True
    n_class = 2
    batchsize = 6
    epochs = 50
    size = 512
    crop_size = 512
    classes = ["0", "1"]
    palette = [[0, 0, 0], [255, 255, 255]]
    gpu = True
    lr = 0.001
    ft = False

    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'0
    model_name = 'ConvNeXt'
    train_dataset = DTTDataSet(is_train=True,
                               data_root='data/train/',
                               img_dir='imgs',
                               ann_dir='masks',
                               classes=classes,
                               palette=palette,
                               size=size,
                               crop_size=crop_size
                               )
    valid_dataset = DTTDataSet(is_train=True,
                               data_root='data/val',
                               img_dir='imgs',
                               ann_dir='masks',
                               classes=classes,
                               palette=palette,
                               size=size,
                               crop_size=crop_size
                               )
    if model_name == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif model_name == 'Res_Unet':
        net = ResUnet(input_channel=3, output_channel=n_class, dropout_rate=0.2)
    elif model_name == 'Ringed_Res_Unet':
        # 分为区域分割和文字分割两种类型
        net = Ringed_Res_Unet(n_channels=3, n_classes=n_class)
        # net_1 = Ringed_Res_Unet(n_channels=3, n_classes=1)
        # net = nn.ModuleList([net_0, net_1])
    elif model_name == "Unet_3Plus":
        net = UNet_3Plus(in_channels=3, n_classes=1)
    elif model_name == "Mvss_Net":
        net = get_mvss(sobel=True, n_input=3, constrain=True)
    elif model_name == "Unet++":
        config = PretrainedConfig.from_json_file('config.json')
        net = HFUnetPlusPlus(config=config)
    elif model_name == "FCN":
        net = FCN16s(n_class=1)
    elif model_name == "ConvNeXt":
        net = nn.Sequential(ConvNeXt(in_chans=3,
                                     depths=[2, 3, 24, 2],
                                     dims=[64, 128, 256, 512],
                                     drop_path_rate=0.4,
                                     layer_scale_init_value=1.0,
                                     out_indices=[0, 1, 2, 3]),
                            UPerHead(in_channels=[64, 128, 256, 512],
                                     in_index=[0, 1, 2, 3],
                                     pool_scales=(1, 2, 3, 6),
                                     channels=512,
                                     dropout_ratio=0.2,
                                     num_classes=n_class,
                                     norm_cfg=dict(type='BN', requires_grad=True),
                                     align_corners=False),
                            )
    # print(net)

    if ft:
        fine_tuning_model = 'result/ConvNeXt-[iou_score]-0.1446-[train_loss]-0.1316.pkl'
        # models_dict = net.state_dict()
        # checkpoint = torch.load(fine_tuning_model)
        # for model in models_dict:
        #     if model in checkpoint:
        #         models_dict[model] = checkpoint[model]
        # print('Checkpoint loaded')
        net.load_state_dict(torch.load(fine_tuning_model))
        print('Model loaded from {}'.format(fine_tuning_model))

    if gpu:
        net.cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory

    train_net(net=net,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              model=model_name,
              epochs=epochs,
              batch_size=batchsize,
              lr=lr,
              gpu=gpu,
              size=size)
