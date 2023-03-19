import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
import numpy as np
from torch import Tensor
from math import exp

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        if input.dim() == 2 and reduce_batch_first:
            raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

        if input.dim() == 2 or reduce_batch_first:
            inter = torch.dot(input.reshape(-1), target.reshape(-1))
            sets_sum = torch.sum(input) + torch.sum(target)
            if sets_sum.item() == 0:
                sets_sum = 2 * inter

            return (2 * inter + epsilon) / (sets_sum + epsilon)
        else:
            # compute and average metric for each batch element
            dice = 0
            for i in range(input.shape[0]):
                dice += self.dice_coeff(input[i, ...], target[i, ...])
            return dice / input.shape[0]

    def multiclass_dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
        # Average of Dice coefficient for all classes
        assert input.size() == target.size()
        dice = 0
        for channel in range(input.shape[1]):
            dice += self.dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

        return dice / input.shape[1]


    def forward(self, input: Tensor, target: Tensor, multiclass: bool = False):
        # Dice loss (objective to minimize) between 0 and 1
        assert input.size() == target.size()
        fn = self.multiclass_dice_coeff if multiclass else self.dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)
    # def __init__(self, weight=None, smooth=1.0):
    #     super(DiceLoss, self).__init__()
    #     self.weight = weight
    #     self.smooth = smooth

    # def forward(self, inputs, targets, mutlti_calss=False):
    #     if mutlti_calss:
    #         num_classes = inputs.size(1)
    #         true_1_hot = torch.eye(num_classes)[targets].to(device=inputs.device, dtype=inputs.dtype)

    #         inputs = torch.softmax(inputs, dim=1)
    #         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

    #         dims = (0, 2, 3)
    #         intersection = torch.sum(inputs * true_1_hot, dims)
    #         cardinality = torch.sum(inputs + true_1_hot, dims)
    #     else:
    #         intersection = torch.sum(inputs, dims)
    #         cardinality = torch.sum(inputs, dims)

    #     dice_scores = (2. * intersection + self.smooth) / (cardinality + self.smooth)

    #     if self.weight is not None:
    #         dice_scores = dice_scores * self.weight

    #     dice_loss = 1 - dice_scores.mean()
    #     return dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, input, target):
#         ce_loss = F.cross_entropy(input, target, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt)**self.gamma * ce_loss
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        # print("sim",sim)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible,
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)

# def _iou(pred, target, size_average = True):

#     b = pred.shape[0]
#     IoU = 0.0
#     for i in range(0,b):
#         #compute the IoU of the foreground
#         Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
#         Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
#         IoU1 = Iand1/Ior1

#         #IoU loss is (1-IoU1)
#         IoU = IoU + (1-IoU1)

#     return IoU/b
def iou_loss(pred, target):
    intersection = (pred & target).float().sum((1, 2))  
    union = (pred | target).float().sum((1, 2)) 
    iou = (intersection + 1e-7) / (union + 1e-7)  
    return 1 - iou.mean()


class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def __call__(self, pred, target):
        return iou_loss(pred, target)

def IOU_score(pred,label):
    iou_out = iou_loss(pred, label)
    # print("iou_loss:", iou_out.data.cpu().numpy())
    return 1 - iou_out


def iou(pred_mask, true_mask):
    """
    计算预测掩码和真实掩码之间的IOU
    Args:
        pred_mask: 预测掩码，二值化的图像
        true_mask: 真实掩码，二值化的图像
    Returns:
        iou_score: IOU得分
    """
    p_mask, t_mask = pred_mask.detach().cpu().numpy(), true_mask.detach().cpu().numpy()
    intersection = np.logical_and(p_mask, t_mask)
    union = np.logical_or(p_mask, t_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def max_iou_at_fdr(pred_masks, true_masks, fdr):
    """
    计算给定错误率下的最大IOU值
    Args:
        pred_masks: 预测掩码列表，每个元素是一个二值化的图像
        true_masks: 真实掩码列表，每个元素是一个二值化的图像
        fdr: 错误率阈值
    Returns:
        max_iou: 最大IOU得分
        threshold: 对应的阈值
    """
    # 计算每个预测掩码的得分
    scores = []
    for pred_mask, true_mask in zip(pred_masks, true_masks):
        score = iou(pred_mask, true_mask)
        scores.append(score)
    
    # 根据错误率阈值计算最大IOU得分
    num_images = len(scores)
    sorted_scores = sorted(scores)
    threshold_index = int(num_images * fdr)
    threshold = sorted_scores[threshold_index]
    max_iou = 0.0
    for score in scores:
        if score >= threshold and score > max_iou:
            max_iou = score
    
    return max_iou, threshold



def dice_loss(output, target, epsilon=1e-6):
    """
    Calculate the Dice Loss.
    Args:
        output: the output from the network (predicted masks)
        target: the ground truth mask
        epsilon: a small constant to avoid division by zero
    Returns:
        the Dice Loss
    """
    # Flatten the output and target masks
    output_flat = output.view(-1)
    target_flat = target.view(-1)

    # Calculate intersection and union
    intersection = (output_flat * target_flat).sum()
    union = output_flat.sum() + target_flat.sum()

    # Calculate Dice Loss
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return 1. - dice

class WeightedDiceLoss(nn.Module):
    def __init__(self, num_classes, weight=None, ignore_index=None, sigmoid_normalization=False):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.ignore_index = ignore_index
        self.sigmoid_normalization = sigmoid_normalization

    def forward(self, inputs, targets):
        N, C, H, W = inputs.size()
        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, C)
        targets = targets.view(-1)

        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]

        # Compute class frequencies
        frequency = torch.zeros(self.num_classes, dtype=torch.float32, device=inputs.device)
        torch.bincount(targets, weights=self.weight, minlength=self.num_classes, out=frequency)

        # Compute class weights
        total_frequency = frequency.sum()
        class_weights = torch.tensor([total_frequency / (frequency[i] * self.num_classes) for i in range(self.num_classes)], dtype=torch.float32, device=inputs.device)

        # Compute probabilities using sigmoid
        if self.sigmoid_normalization:
            inputs = torch.sigmoid(inputs)

        # Compute one-hot targets
        targets_one_hot = torch.zeros((targets.size(0), self.num_classes), dtype=torch.float32, device=inputs.device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Compute intersection and union
        intersection = (inputs * targets_one_hot).sum(dim=0)
        union = (inputs + targets_one_hot).sum(dim=0) - intersection

        # Compute dice loss
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss *= class_weights
        dice_loss = dice_loss.sum()

        return dice_loss
    
# class TverskyLoss(nn.Module):
#     def __init__(self, alpha=0.5, beta=0.5, num_classes=2, epsilon=1e-6):
#         super(TverskyLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.num_classes = num_classes
#         self.epsilon = epsilon
        
#     def forward(self, y_pred, y_true):
#         # convert y_true to one-hot tensor
#         y_true = F.one_hot(y_true, self.num_classes).float()
        
#         # compute TP, FN, and FP for each class
#         TP = torch.sum(y_true * y_pred, dim=[0, 2, 3])
#         FN = torch.sum(y_true * (1 - y_pred), dim=[0, 2, 3])
#         FP = torch.sum((1 - y_true) * y_pred, dim=[0, 2, 3])
        
#         # compute Tversky index for each class
#         Tversky = TP / (TP + self.alpha * FN + self.beta * FP + self.epsilon)
        
#         # compute Tversky loss as 1 - Tversky index
#         Tversky_loss = 1 - torch.mean(Tversky)
        
#         return Tversky_loss
    
def tversky_loss(logits, labels, alpha=0.8, beta=0.5):
    """
    Tversky Loss implementation for binary segmentation
    :param logits: output from model
    :param labels: ground truth labels
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :return: tversky loss
    """
    prob = logits
    true_pos = (prob * labels).sum(dim=(1, 2))  # Compute true positives
    false_pos = (prob * (1 - labels)).sum(dim=(1, 2))  # Compute false positives
    false_neg = ((1 - prob) * labels).sum(dim=(1, 2))  # Compute false negatives
    tversky_index = true_pos / (true_pos + alpha * false_pos + beta * false_neg)  # Compute Tversky Index
    tversky_loss = 1 - tversky_index  # Compute Tversky Loss
    return tversky_loss.mean()