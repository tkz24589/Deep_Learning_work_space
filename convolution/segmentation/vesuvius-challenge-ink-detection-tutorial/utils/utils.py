import numpy as np
import cv2
from tqdm import tqdm
from config.config import CFG
import random
import torch
import torch.nn as nn

def read_image_mask(fragment_id, tile_size):

    images = []
    idxs = CFG.in_idx

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
    
    return images, mask, mask_location

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
        
        if fragment_id == 1:
            tile_size = CFG.train_tile_size_1
            stride = CFG.train_stride_1
        elif fragment_id == 2:
            tile_size = CFG.train_tile_size_2
            stride = CFG.train_stride_2
        else:
            tile_size = CFG.train_tile_size_3
            stride = CFG.train_stride_3

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
        
                if fragment_id == CFG.valid_id:
                    if CFG.valid_id  == 2:
                        if  y2 <4800 or y2 > 4800 + 4096 + 2048 or x2 > 640+ 4096 +2048 or x2 < 640:
                            continue
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def TTA(x:torch.Tensor, model:nn.Module):
    shape = x.shape
    x = [ x,*[torch.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)]]
    x = torch.cat(x,dim=0)
    x = model(x)
    x = x.reshape(4,shape[0],1,*shape[-2:])
    x = [torch.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
    x = torch.stack(x,dim=0)
    x = torch.sigmoid(x)
    return x.mean(0)

# Copyright (c) OpenMMLab. All rights reserved.
import functools

import mmcv
import numpy as np
import torch.nn.functional as F


def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            # pkl, json or yaml
            class_weight = mmcv.load(class_weight)

    return class_weight


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper
