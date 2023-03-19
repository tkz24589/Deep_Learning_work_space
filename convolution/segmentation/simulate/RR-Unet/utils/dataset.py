import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image

from utils.test_time_aug import MultiScaleFlipAug
from .utils import SquarePad, SquarePadImg
import cv2
# from torchvision.transforms import Normalize
from torchvision import transforms as tfs
from utils import transforms, formatting, loading, custom
from utils.transforms import Resize, RandomCrop, RandomFlip, RandomRotate90, PhotoMetricDistortion, Normalize, Pad

class DTTDataSet(data.Dataset):
    def __init__(self, is_train, data_root, img_dir, ann_dir, classes, palette, size=512, crop_size=1024) -> None:
        super(DTTDataSet, self).__init__()
        self.is_train = is_train
        self.crop_size = crop_size
        self.size = size
        self.dataset = custom.CustomDataset(data_root=data_root,
                                            img_dir=img_dir,
                                            ann_dir=ann_dir,
                                            img_suffix=".jpg",
                                            seg_map_suffix='.png',
                                            classes=classes,
                                            palette=palette,
                                            use_mosaic=False,
                                            mosaic_prob=0.5)
        self.transform_train = transforms.Compose([
            loading.LoadImageFromFile(),
            loading.LoadAnnotations(),
            Resize(img_scale=(size, size), keep_ratio=False),
            # RandomCrop(crop_size=(crop_size, crop_size), cat_max_ratio=1.0),
            RandomFlip(prob=0.5, direction='horizontal'),
            RandomFlip(prob=0.5, direction='vertical'),
            RandomRotate90(prob=0.5),
            PhotoMetricDistortion(),
            Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            Pad(size=(crop_size, crop_size), pad_val=0, seg_pad_val=255),
            formatting.DefaultFormatBundle(),
            formatting.Collect(keys=['img', 'gt_semantic_seg'])
        ])
        self.transform_val = transforms.Compose([
            loading.LoadImageFromFile(),
            loading.LoadAnnotations(),
            # Resize(img_scale=(size, size), keep_ratio=True),
            # RandomCrop(crop_size=(crop_size, crop_size), cat_max_ratio=0.9),
            RandomFlip(prob=0.5, direction='horizontal'),
            RandomFlip(prob=0.5, direction='vertical'),
            RandomRotate90(prob=0.5),
            PhotoMetricDistortion(),
            Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            formatting.DefaultFormatBundle(),
            formatting.Collect(keys=['img', 'gt_semantic_seg'])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.is_train:
            result  = self.transform_train(self.dataset[index])
        else:
            result  = self.transform_train(self.dataset[index])
        img = result['img'].data
        mask = result['gt_semantic_seg'].data
        # info = result['img_metas']
        # m = mask.numpy()
        mask = torch.where(mask < 127, torch.Tensor([0.]), torch.Tensor([1.])).long()
        flag = 1
        if torch.sum(mask) == 0:
            flag = 0
        return img, mask, flag


