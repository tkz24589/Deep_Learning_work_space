import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import segmentation_models_pytorch as smp
from torch.cuda import amp
from torch.optim import lr_scheduler
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import gc
from PIL import Image
import time
import copy
from collections import defaultdict
from colorama import Fore, Style
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
c_ = Fore.GREEN
sr_ = Style.RESET_ALL


class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    learning_rate = 0.001
    weight_decay = 0.0001
    epochs = 10
    n_accumulate = batch_size
    scheduler = 'CosineAnnealingLR'
    T_max = int(30000 / batch_size * epochs) + 50
    min_lr = 1e-6
    T_0 = 25
    img_size = [512,512]
    model_name = 'ResUnet'


# 测试图片
# test_img = r'E:\TianChi\Image_Compition\Text_Manipulation_Detection\train\untampered\imgs\0074.jpg'
# '''Data Preparation'''

# 数据源
# data_sources = r'data/train/tampered/imgs'
# mask_sources = r'data/train/tampered/masks'
# sources = {'data': data_sources, 'mask': mask_sources}

# train_character_txt = r"data/train/final_pure_character.txt"  # 1300
# val_datasets_txt = r"data/train/impurity_img_after_classify.txt"  # 600左右

# negative_samples = r"E:\TianChi\Image_Compition\Text_Manipulation_Detection\train\untampered\imgs"  # 选1900
# negative_masks_path = r"E:\TianChi\Image_Compition\Text_Manipulation_Detection\train\untampered\masks"

# 数据集
'''datasets'''


def make_datasets(sources, list_txt):
    f = open(list_txt, 'r')
    for name in f.readlines():
        name = name.split('.')[0]
        img = sources['data'] + os.sep + name + '.jpg'
        mask = sources['mask'] + os.sep + name + '.png'
        yield (img, mask)


def make_mask(img, negative_masks_path):
    mask_name = str(img.name).split('.')[0] + '.png'
    mask_path = negative_masks_path + os.sep + mask_name
    img = str(img)
    mask = cv2.imread(img).copy()
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = mask * 0
    cv2.imwrite(mask_path, mask)
    return


def make_positive_datasets(negative_path, negative_masks_path):
    path = Path(negative_path)
    negative_imgs = path.glob('*.jpg')
    ls = []
    for img in negative_imgs:
        # make_mask(img, negative_masks_path)
        img_name = img.name
        mask_name = img_name.split('.')[0] + '.png'
        mask = negative_masks_path + os.sep + mask_name
        ls.append((str(img), mask))
    return ls


# train_ds = make_datasets(sources, train_character_txt)
# val_ds = make_datasets(sources, val_datasets_txt)
# positive_ds = make_positive_datasets(negative_samples, negative_masks_path)

# train_ds = list(iter(train_ds))
# val_ds = list(iter(val_ds))

# train_ds.extend(positive_ds[:1300])
# val_ds.extend(positive_ds[1300:1900])

'''transforms'''


class SquarePadImg:
    def __call__(self, img):
        w, h = img.size
        max_size = max(w, h)
        pad_h = (max_size - h) // 2
        pad_w = (max_size - w) // 2
        padding = (pad_w, pad_h, max_size - pad_w - w, max_size - pad_h - h)
        return transforms.functional.pad(img, padding, 0, 'constant')


data_transforms = {
    'img': transforms.Compose([
        transforms.ToTensor(),
        SquarePadImg(),
        transforms.Resize([512, 512]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'mask': transforms.Compose([
        transforms.ToTensor(),
        SquarePadImg(),
        transforms.Resize([512, 512]),
    ]),
    'val': transforms.Compose([
        SquarePadImg(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ])
}


class MyLazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.dataset[index][0])
        mask = Image.open(self.dataset[index][1])
        if self.transform:
            img = self.transform['img'](img)
            mask = self.transform['mask'](mask)

        return img, mask

    def __len__(self):
        return len(self.dataset)
from utils.dataset import DTTDataSet

train_ds = DTTDataSet(is_train=True,
                               data_file="data/train/train_by_hand.npz",
                               size=CFG.img_size[0],
                               )
val_ds = DTTDataSet(is_train=False,
                            data_file="data/train/valid_by_hand.npz",
                            size=CFG.img_size[0],
                            )

# train_ds = DTTDataSet(train_ds, data_transforms)
# val_ds = DTTDataSet(val_ds, data_transforms)

train_loader = DataLoader(dataset=train_ds, batch_size=CFG.batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(dataset=val_ds, batch_size=CFG.batch_size, shuffle=False, pin_memory=True, num_workers=4)


# Models

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.identity_map = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = inputs.clone().detach()  # detach()不参与梯度下降的意思好像
        out = self.layer(x)
        residual = self.identity_map(inputs)
        skip = out + residual
        return self.relu(skip)


class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels)
        )

    def forward(self, inputs):
        return self.layer(inputs)


class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block = ResBlock(in_channels + out_channels, out_channels)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x


class ResUnet(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_rate=0.2):
        super().__init__()
        self.encoding_layer1_ = ResBlock(input_channel, 64)
        self.encoding_layer2_ = DownSampleConv(64, 128)
        self.encoding_layer3_ = DownSampleConv(128, 256)
        self.bridge = DownSampleConv(256, 512)
        self.decoding_layer3_ = UpSampleConv(512, 256)
        self.decoding_layer2_ = UpSampleConv(256, 128)
        self.decoding_layer1_ = UpSampleConv(128, 64)
        self.output = nn.Conv2d(64, output_channel, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, inputs):
        ###################### Enocoder #########################
        e1 = self.encoding_layer1_(inputs)
        e1 = self.dropout(e1)
        e2 = self.encoding_layer2_(e1)
        e2 = self.dropout(e2)
        e3 = self.encoding_layer3_(e2)
        e3 = self.dropout(e3)

        ###################### Bridge #########################
        bridge = self.bridge(e3)
        bridge = self.dropout(bridge)

        ###################### Decoder #########################
        d3 = self.decoding_layer3_(bridge, e3)
        d2 = self.decoding_layer2_(d3, e2)
        d1 = self.decoding_layer1_(d2, e1)

        ###################### Output #########################
        output = self.output(d1)
        return output



'''Loss Function'''
JaccardLoss = smp.losses.JaccardLoss(mode='binary')
DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode='binary', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def criterion(y_pred, y_true):
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)


'''Traning Function'''


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train')
    for step, (images, masks, _) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss = loss / CFG.n_accumulate

        scaler.scale(loss).backward()

        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


'''Validation Function'''


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks, _) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, masks)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')
    val_scores = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_scores


'''Run Training'''


def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients


    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler,
                                     dataloader=train_loader,
                                     device=device, epoch=epoch)

        val_loss, val_scores = valid_one_epoch(model, val_loader,
                                               device=device,
                                               epoch=epoch)
        val_dice, val_jaccard = val_scores

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)

        # Log the metrics

        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            torch.save(model.state_dict(),
                "result/" + '{}-[iou_score]-{:.4f}-[dice_score]-{:.4f}-[train_loss]-{:.4f}-[valid_loss]-{:.4f}.pkl'.format("DDT", val_jaccard, val_dice, train_loss, val_loss))

            print(f"Model Saved{sr_}")

    torch.save(model.state_dict(),
            "result/" + '{}-[iou_score]-{:.4f}-[dice_score]-{:.4f}-[train_loss]-{:.4f}-[valid_loss]-{:.4f}.pkl'.format("DDT_last_model", val_jaccard, val_dice, train_loss, val_loss))


    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_jaccard))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


'''Optimizer'''


def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max,
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0,
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr, )
    elif CFG.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler == None:
        return None

    return scheduler


model = ResUnet(3, 1).to(CFG.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
scheduler = fetch_scheduler(optimizer)

'''Training'''
for fold in range(CFG.epochs):
    print(f'#' * 15)
    print(f'### Fold: {fold}')
    print(f'#' * 15)

    # model = ResUnet(3, 1).to(CFG.device)
    # train_loader = DataLoader(dataset=train_ds, batch_size=CFG.batch_size, shuffle=True, pin_memory=True)
    # val_loader = DataLoader(dataset=val_ds, batch_size=CFG.batch_size, shuffle=False, pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = fetch_scheduler(optimizer)
    model, history = run_training(model, optimizer, scheduler,
                                  device=CFG.device,
                                  num_epochs=CFG.epochs)


'''Prediction'''


# from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation

# extractor = AutoFeatureExtractor.from_pretrained("jonathandinu/face-parsing")
# model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
# print(model)
# img = Image.open(test_img)
# inputs = extractor(img, return_tensors='pt')
#
# with torch.no_grad():
#     outputs = model(**inputs)
#
#
# predicted_label = outputs.logits    # (1, 19, 128, 128)
# array1 = predicted_label.numpy()
# maxValue=array1.max()
# array1=array1*255/maxValue  # normalize，将图像数据扩展到[0,255]
# mat=np.uint8(array1)    # float32-->uint8
# print('mat_shape:',mat.shape)   # mat_shape: (3, 982, 814)
# mat=mat.transpose(1,2,0)    # mat_shape: (982, 814，3)
# cv2.imshow("img",mat)
# cv2.waitKey()


# res = model(inputs)
# print(res)
