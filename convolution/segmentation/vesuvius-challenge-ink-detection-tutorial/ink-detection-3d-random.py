#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
from torchsummary import summary
from einops.layers.torch import Rearrange
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch.cuda.amp import GradScaler
from torchvision.utils import make_grid
import warnings
import os
import random
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
# 忽略所有警告
warnings.filterwarnings('ignore')
gc.collect()
torch.cuda.empty_cache()

class CFG:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    checpoint = 'result/unet/unet++/30/UnetPlusPlus-30.pkl'
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    comp_dir_path = ''
    comp_folder_name = '../Datasets/vesuvius-challenge-ink-detection/'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    img_path = 'working/'
    exp_name = 'Resnet3D'

    # ============== pred target =============
    target_size = 1
    # ============== model cfg =============
    model_name = 'Resnet3D'

    in_idx = [i for i in range(12, 25)]
    up = [i for i in range(28, 37)]

    in_idx.extend(up)

    backbone = 'Resnet3D'

    valid_id = "random"


    in_chans = 32# 65
    # ============== training cfg =============
    size = 224

    train_tile_size_1 = 224
    train_stride_1 = train_tile_size_1 // 5

    train_tile_size_2 = 224
    train_stride_2 = train_tile_size_2 // 3

    train_tile_size_3 = 224
    train_stride_3= train_tile_size_3 // 5

    valid_tile_size = 224
    valid_stride = valid_tile_size // 2

    buffer = 32

    rate_valid = 0.05

    train_batch_size = 22 # 32
    valid_batch_size = 22
    use_amp = True

    epochs = 30 # 30

    # lr = 1e-4 / warmup_factor
    lr = 1e-5

    # ============== fixed =============
    pretrained = False

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000

    num_workers = 4

    seed = 42

    threshhold = 0.5

    all_best_dice = 0
    all_best_loss = np.inf

    shape_list = []
    test_shape_list = []

    val_mask = None
    val_label = None

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.8, 1.5)),
        A.Resize(size, size),
        A.RandomRotate90(always_apply=False,p=0.2),
        # A.RandomGridShuffle(grid=(4, 4), always_apply=False, p=0.2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    test_aug_list = [
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
seed = CFG.seed
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[2]:


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 shortcut_type='B',
                 widen_factor=1.0):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block,
                                       block_inplanes[0],
                                       layers[0],
                                       shortcut_type,
                                       stride=(1, 1, 1),
                                       downsample=False)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=(1, 2, 2),
                                       downsample=True)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=(1, 2, 2),
                                       downsample=True)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=(1, 2, 2),
                                       downsample=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride, downsample):
        downsample_block = None
        if downsample:
            if shortcut_type == 'A':
                downsample_block = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample_block = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample_block))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

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

        return [x1, x2, x3, x4]


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

# class ResBlock2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1, bias=bias),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.identity_map = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, inputs):
#         x = inputs.clone().detach()
#         out = self.layer(x)
#         residual = self.identity_map(inputs)
#         skip = out + residual
#         return self.relu(skip)

class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = generate_model(model_depth=18, n_input_channels=1)
        self.decoder = Decoder(encoder_dims=[64, 128, 256, 512], upscale=4)
        
    def forward(self, x):
        feat_maps = self.encoder(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask
    
    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        print(self.encoder.load_state_dict(state_dict, strict=False))


# In[3]:

model = SegModel()
model.load_pretrained_weights(torch.load("result/resnet3d-seg-2d/r3d18_K_200ep.pth")["state_dict"])
model = nn.DataParallel(model, device_ids=[0])
model = model.to(CFG.device)
model_name = CFG.model_name
if CFG.pretrained:
    try:
        checkpoint = torch.load(CFG.checpoint, map_location=CFG.device)
        models_dict = model.state_dict()
        for model_part in models_dict:
            if model_part in checkpoint:
                models_dict[model_part] = checkpoint[model_part]
        model.load_state_dict(models_dict)
        print('Checkpoint loaded')
    except:
        print('Checkpoint not loaded')
        pass


# In[4]:


def read_image_mask(fragment_id, tile_size):

    images = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)

    for i in tqdm(idxs):
        
        image = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)

        pad0 = (tile_size - image.shape[0] % tile_size)
        pad1 = (tile_size - image.shape[1] % tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)

    mask = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype('float32')
    mask /= 255.0
    
    mask_location = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/mask.png", 0)
    mask_location = np.pad(mask_location, [(0, pad0), (0, pad1)], constant_values=0)

    mask_location = mask_location / 255
    CFG.shape_list.append(mask.shape)
    
    return images, mask, mask_location


# In[5]:
def get_random_train_valid_dataset(rate_valid=0.05):
    images = []
    masks = []
    xyxys = []
    for fragment_id in range(1, 4):
        if fragment_id == 1:
            tile_size = CFG.train_tile_size_1
            stride = CFG.train_stride_1
        elif fragment_id == 2:
            tile_size = CFG.train_tile_size_2
            stride = CFG.train_stride_2
        else:
            tile_size = CFG.train_tile_size_3
            stride = CFG.train_stride_3

        image, mask, mask_location = read_image_mask(fragment_id, tile_size)
        x1_list = list(range(0, image.shape[1] - tile_size + 1, stride))
        y1_list = list(range(0, image.shape[0] - tile_size + 1, stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + tile_size
                x2 = x1 + tile_size
                if np.sum(mask_location[y1:y2, x1:x2]) == 0:
                    continue
                images.append(image[y1:y2, x1:x2])
                masks.append(mask[y1:y2, x1:x2, None])
                xyxys.append([x1, y1, x2, y2])

    '''random images&masks get train&valid'''
    images_masks_list = list(zip(images, masks, xyxys))
    random.shuffle(images_masks_list)
    images[:], masks[:], xyxys[:] = zip(*images_masks_list)
    valid_numbers = int(len(images)*rate_valid)

    train_images = images[:-valid_numbers]
    train_masks = masks[:-valid_numbers]
    valid_images = images[-valid_numbers:]
    valid_masks = masks[-valid_numbers:]
    valid_xyxys = xyxys[-valid_numbers:]

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys



def get_train_valid_dataset():
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in range(1, 4):

        # valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/mask.png", 0)
        # valid_mask_gt = valid_mask_gt / 255
        # pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
        # pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
        # valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
        if fragment_id == 1:
            tile_size = CFG.train_tile_size_1   # 224
            stride = CFG.train_stride_1    # 56
        elif fragment_id == 2:
            tile_size = CFG.train_tile_size_2   # 224
            stride = CFG.train_stride_2    # 112
        else:
            tile_size = CFG.train_tile_size_3   # 224
            stride = CFG.train_stride_3    # 56

        if fragment_id == CFG.valid_id:
            tile_size = CFG.valid_tile_size
            stride = CFG.valid_stride
            
        image, mask, mask_location = read_image_mask(fragment_id, tile_size)
        x1_list = list(range(0, image.shape[1]-tile_size+1, stride))
        y1_list = list(range(0, image.shape[0]-tile_size+1, stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + tile_size
                x2 = x1 + tile_size
                if np.sum(mask_location[y1:y2, x1:x2]) == 0:
                    continue
                # if np.sum(valid_mask_gt[y1:y2, x1:x2]) < CFG.stride * CFG.stride / 10:
                #     continue
                # xyxys.append((x1, y1, x2, y2))
        
                if fragment_id == CFG.valid_id:
                    # if CFG.valid_id  == 2:
                    #     if  y2 <4800 or y2 > 4800 + 4096 + 2048 or x2 > 640+ 4096 +2048 or x2 < 640:
                    #         continue
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


# In[6]:


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    # print(aug)
    return aug

class Ink_Detection_Dataset(data.Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image'].unsqueeze(0)
            label = data['mask']

        return image, label


# In[7]:

if CFG.valid_id == "random":
    train_images, train_masks, valid_images, valid_masks, _ = get_random_train_valid_dataset(CFG.rate_valid)
else:
    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()
    valid_xyxys = np.stack(valid_xyxys)

# In[8]:





# In[9]:


train_dataset = Ink_Detection_Dataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
valid_dataset = Ink_Detection_Dataset(
    valid_images, CFG, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

train_loader = data.DataLoader(train_dataset,
                          batch_size=CFG.train_batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=False, drop_last=True,
                          )
valid_loader = data.DataLoader(valid_dataset,
                          batch_size=CFG.valid_batch_size,
                          shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=False, drop_last=False)


# In[10]:


from warmup_scheduler import GradualWarmupScheduler


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

# def dice_coef(y_true, y_pred, thr=0.5, dim=(0, 1), epsilon=0.001):
#     y_true = y_true.to(torch.float32)
#     y_pred = (y_pred > thr).to(torch.float32)
#     inter = (y_true * y_pred).sum(dim=dim)
#     den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
#     dice = ((2 * inter + epsilon) / (den + epsilon)).mean()
#     return dice



def iou_coef(y_true, y_pred, thr=0.5, dim=(0, 1), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean()
    return iou

def dice_coef(targets, preds, thr=0.5, beta=0.5, smooth=1e-5):

    #comment out if your model contains a sigmoid or equivalent activation layer
    # flatten label and prediction tensors
    preds = (preds > thr).view(-1).float()
    targets = targets.view(-1).float()

    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


# In[11]:


def train_step(train_loader, model, criterion, optimizer, writer, device, epoch):
    model.train()
    epoch_loss = 0
    scaler = GradScaler(enabled=CFG.use_amp)
    bar = tqdm(enumerate(train_loader), total=len(train_loader)) 
    for step, (image, label) in bar:
        optimizer.zero_grad()
        outputs = model(image.to(device))
        loss = criterion(outputs, label.to(device))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{loss.item():0.4f}', epoch=epoch ,gpu_mem=f'{mem:0.2f} GB', lr=f'{optimizer.state_dict()["param_groups"][0]["lr"]:0.2e}')
        epoch_loss += loss.item()
    writer.add_scalar('Train/Loss', epoch_loss / len(train_loader), epoch)
    return epoch_loss / len(train_loader)

def valid_step(valid_loader, model, criterion, writer, device, epoch):

    model.eval()
    epoch_loss = 0
    count = 0
    dice_scores = []
    iou_scores = []

    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, (images, labels) in bar:
        images = images.to(device)
        labels = labels.to(device)


        with torch.no_grad():
            y_preds = model(images)
            loss = criterion(y_preds, labels)
        # make whole mask
        y_preds = torch.sigmoid(y_preds)

        epoch_loss += loss.item()
        count = count + 1
        # 计算准确率
        dice_score = dice_coef(labels.to(device), y_preds.to(device), thr=CFG.threshhold).item()
        iou_socre = iou_coef(labels.to(device), y_preds.to(device), thr=CFG.threshhold).item()

        dice_scores.append(dice_score)
        iou_scores.append(iou_socre)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{epoch_loss/count:0.4f}', epoch=epoch, gpu_mem=f'{mem:0.2f} GB')


    # writer.add_scalar('Val/IOU', sum(iou_scores) / len(iou_scores), epoch)
    # writer.add_scalar('Val/Dice', sum(dice_scores) / len(dice_scores), epoch)
    # writer.add_scalar('Valid/Loss', epoch_loss / len(valid_loader), epoch)
    torch.save(model.state_dict(),
               'result/' + '{}-DIM-{}-[eval_loss]-{:.4f}-[dice_score]-{:.2f}-[iou_score]-{:.2f}-'.format(model_name,
                                                                                                         CFG.in_chans,
                                                                                                         epoch_loss / len(
                                                                                                             valid_loader),
                                                                                                         sum(dice_scores) / len(
                                                                                                             dice_scores),
                                                                                                         sum(iou_scores) / len(
                                                                                                             iou_scores)) + str(
                   epoch) + '-epoch.pkl')
# def valid_step(valid_loader, model, valid_xyxys, valid_mask_gt , criterion, device, writer, epoch):
#     model.eval()
#     mask_pred = np.zeros(valid_mask_gt.shape)
#     mask_count = np.zeros(valid_mask_gt.shape)
#     valid_mask_gt = np.zeros(valid_mask_gt.shape)
#
#     epoch_loss = 0
#     dice_scores = {}
#     for th in np.arange(1, 6, 0.5) / 10:
#         dice_scores[th] = []
#
#     bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
#     for step, (image, label) in bar:
#         image = image.to(device)
#         label = label.to(device)
#         with torch.no_grad():
#             y_pred = model(image)
#             loss = criterion(y_pred, label)
#         mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
#         bar.set_postfix(loss=f'{loss.item():0.4f}', epoch=epoch ,gpu_mem=f'{mem:0.2f} GB')
#         # make whole mask
#         y_pred = torch.sigmoid(y_pred).to('cpu').numpy()
#         label = label.to('cpu').numpy()
#         start_idx = step*CFG.valid_batch_size
#         end_idx = start_idx + CFG.valid_batch_size
#         for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
#             mask_pred[y1:y2, x1:x2] += y_pred[i].squeeze(0)
#             valid_mask_gt[y1:y2, x1:x2] = label[i].squeeze(0)
#             mask_count[y1:y2, x1:x2] += np.ones((CFG.valid_tile_size, CFG.valid_tile_size))
#         epoch_loss += loss.item()
#     avg_loss = epoch_loss / len(valid_loader)
#     writer.add_scalar('Valid/Loss', avg_loss, epoch)
#     best_th = 0
#     best_dice = 0
#     print(f'mask_count_min: {mask_count.min()}')
#     mask_pred /= mask_count
#     # if CFG.valid_id == 2:
#     #     # 防止内存溢出if  y2 <4800 or y2 > 4800 + 4096 + 2048 or x2 > 640+ 4096 +2048 or x2 < 640:
#     #     valid_mask_gt = valid_mask_gt[4800:4800+4096+2048, 640:640+4096+2048]
#     #     mask_pred = mask_pred[4800:4800+4096+2048, 640:640+4096+2048]
#     for th in np.arange(1, 6, 0.5) / 10:
#         dice_score = dice_coef(torch.from_numpy(valid_mask_gt).to(CFG.device), torch.from_numpy(mask_pred).to(CFG.device), thr=th).item()
#         dice_scores[th].append(dice_score)
#     for th in np.arange(1, 6, 0.5) / 10:
#         dice_score = sum(dice_scores[th]) / len(dice_scores[th])
#         if dice_score > best_dice:
#             best_dice = dice_score
#             best_th = th
#     # # 使用make_grid将图片转换成网格形式
#     # pred_mask = make_grid((torch.from_numpy(mask_pred) > best_th).float().to(CFG.device), normalize=True)
#     # true_mask = make_grid(torch.from_numpy(valid_mask_gt).to(CFG.device), normalize=True)
#     # # 使用add_image方法将图片添加到TensorBoard中
#     # writer.add_image('Valid/True_mask', true_mask, global_step=epoch, dataformats="CHW")
#     # writer.add_image('Valid/Pred_mask', pred_mask, global_step=epoch, dataformats="CHW")
#     mask_pred = (mask_pred >= best_th).astype(int)
#     cv2.imwrite(f'result/logs/{epoch}.png', mask_pred * 255)
#     cv2.imwrite(f'result/logs/gt.png', valid_mask_gt * 255)
#     if CFG.all_best_dice < best_dice:
#         print('best_th={:2f}' .format(best_th),"score up: {:2f}->{:2f}".format(CFG.all_best_dice, best_dice))
#         CFG.all_best_dice = best_dice
#     torch.save(model.state_dict(), 'result/' +  '{}-DIM-{}-[eval_loss]-{:.4f}-[dice_score]-{:.2f}-'.format(model_name, CFG.in_chans , avg_loss, best_dice) + str(epoch) + '-epoch.pkl')
#     writer.add_scalar('Valid/Dice', best_dice, epoch)
#
#     return avg_loss
    


# In[12]:


# fragment_id = CFG.valid_id
# valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
# valid_mask_gt = valid_mask_gt / 255
# pad0 = (CFG.valid_tile_size - valid_mask_gt.shape[0] % CFG.valid_tile_size)
# pad1 = (CFG.valid_tile_size - valid_mask_gt.shape[1] % CFG.valid_tile_size)
# valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)


# In[13]:


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(),
                        lr=CFG.lr,
                        betas=(0.9, 0.999),
                        weight_decay=CFG.weight_decay
                        )
scheduler = get_scheduler(CFG, optimizer)
writer = SummaryWriter('result/logs')
for i in range(CFG.epochs):
    print('train:')
    train_step(train_loader, model, criterion, optimizer, writer, CFG.device, i + 1)
    print('val:')
    valid_step(valid_loader, model, criterion, writer, CFG.device, i + 1)
# for i in range(CFG.epochs):
#     print('train:')
#     train_step(train_loader, model, criterion, optimizer, writer, CFG.device, i + 1)
#     print('val:')
#     val_loss = valid_step(valid_loader, model, valid_xyxys, valid_mask_gt, criterion, CFG.device, writer,  i + 1)
#     scheduler_step(scheduler, val_loss, i + 1)
#     gc.collect()


# In[ ]:




