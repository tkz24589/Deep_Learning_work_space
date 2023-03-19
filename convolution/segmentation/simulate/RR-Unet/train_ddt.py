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
import numpy as np
import time
import copy
from collections import defaultdict
from colorama import Fore, Style
import warnings
import json

warnings.filterwarnings('ignore')
c_ = Fore.GREEN
sr_ = Style.RESET_ALL


class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 6
    learning_rate = 0.001
    weight_decay = 0.0001
    epochs = 10
    fords = 1
    n_accumulate = 1
    scheduler = 'CosineAnnealingLR'
    T_max = int(30000 / batch_size * epochs) + 50
    min_lr = 1e-6
    T_0 = 25
    img_size = [512, 512]
    model_name = 'ResUnet'
    rate_train = 0.90
    rate_val = 0.10
    rate_test = 0.00


# 测试图片
test_img = r'data/train/tampered/imgs/0000.jpg'
test_mask = r'data/train/tampered/masks/0000.png'
# '''Data Preparation'''

# 数据源
data_sources = r'data/train/tampered/imgs'
mask_sources = r'data/train/tampered/masks'
sources = {'data': data_sources, 'mask': mask_sources}

# train_character_txt = r"E:\TianChi\Image_Compition\tools\final_pure_character.txt"  # 1300
positive_datasets_txt = r"data/train/positive_imgs.txt"  # 3999
# val_datasets_txt = r"E:\TianChi\Image_Compition\tools\impurity_img_after_classify.txt"  # 600左右

negative_samples = r"E:\TianChi\Image_Compition\Text_Manipulation_Detection\train\untampered\imgs"  # 选4000
negative_masks_path = r"E:\TianChi\Image_Compition\Text_Manipulation_Detection\train\untampered\masks"

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


def make_negative_datasets(negative_path, negative_masks_path):
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


positive_ds = make_datasets(sources, positive_datasets_txt)
# val_ds = make_datasets(sources, val_datasets_txt)
negative_ds = make_negative_datasets(negative_samples, negative_masks_path)

positive_ds = list(iter(positive_ds))
positive_ds.extend(negative_ds[:4000])
datasets = positive_ds
num_datasets = len(datasets)
num_train = int(num_datasets * CFG.rate_train)
num_val = int(num_datasets * CFG.rate_val)
num_test = int(num_datasets * CFG.rate_test)
train_ds, val_ds, test_ds = torch.utils.data.random_split(datasets, [num_train, num_val, num_test])

'''transforms'''


class SquarePad2:
    def __call__(self, img):
        _, h, w = img.shape
        max_size = max(w, h)
        pad_h = (max_size - h) // 2
        pad_w = (max_size - w) // 2
        padding = (pad_w, max_size - pad_w - w, pad_h, max_size - pad_h - h)
        return F.pad(img, padding, 'constant', 0)


class Divide:
    def __init__(self, x):
        self.x = x

    def __call__(self, img):
        return torch.div(img, self.x)


data_transforms = {
    'img': transforms.Compose([
        transforms.ToTensor(),
        SquarePad2(),
        Divide(255),
        transforms.Resize(CFG.img_size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'mask': transforms.Compose([
        transforms.ToTensor(),
        SquarePad2(),
        Divide(255),
        transforms.Resize(CFG.img_size),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        SquarePad2(),
        Divide(255),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        SquarePad2(),
        transforms.Resize(CFG.img_size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}
##test
# test_img = Image.open(test_mask)
# test_img = data_transforms['mask'](test_img)
# print(test_img)
# show_img = transforms.ToPILImage()(test_img)
# show_img.show()


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
            mask_c = mask.cpu().numpy()

        return img, mask

    def __len__(self):
        return len(self.dataset)


train_ds = MyLazyDataset(train_ds, data_transforms)
val_ds = MyLazyDataset(val_ds, data_transforms)

train_loader = DataLoader(dataset=train_ds, batch_size=CFG.batch_size, shuffle=True, pin_memory=True, num_workers=2)
val_loader = DataLoader(dataset=val_ds, batch_size=CFG.batch_size, shuffle=False, pin_memory=True, num_workers=2)


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


def dice_coef(y_true, y_pred, thr=0.9, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.9, dim=(2, 3), epsilon=0.001):
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

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:
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
def valid_one_epoch(model, dataloader, device, optimizer):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:
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
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler,
                                     dataloader=train_loader,
                                     device=device, epoch=epoch)

        val_loss, val_scores = valid_one_epoch(model, val_loader,
                                               device=device,
                                               optimizer=optimizer)
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
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"./models/best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory

            print(f"Model Saved{sr_}")

        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"./models/last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), PATH)

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

'''Training'''
def train():
    model = ResUnet(3, 1).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = fetch_scheduler(optimizer)
    f = open('history.json', 'w')
    for fold in range(CFG.fords):  # 这里不应该是CFG.epochs，而是一大轮，自己设定
            print(f'#' * 15)
            print(f'### Fold: {fold}')
            print(f'#' * 15)
            model, history = run_training(model, optimizer, scheduler,
                                        device=CFG.device,
                                        num_epochs=CFG.epochs)

    '''Prediction'''
import numpy as np
from utils import *
from utils.data_vis import plot_img_and_mask


def predict_img(net,
                img_org,
                full_img,
                net_name,
                slice=256,
                size=512,
                stride=128,
                is_slice=False,
                out_threshold=0.5,
                use_gpu=True):
    net.eval()

    if use_gpu:
        full_img = full_img.cuda().unsqueeze(0)

    with torch.no_grad():
        mask = net(full_img)
        mask = nn.Sigmoid()(mask)
        mask_pred = (mask.squeeze() > 0.9).int().cpu().numpy()

        # 裁剪图片到原大小
        padding = get_padding(img_org)
        box = (padding[1], padding[0], padding[1] + img_org.width, padding[0] + img_org.height)
        max_l = max(img_org.width, img_org.height)
        mask_pred = mask_to_image(mask_pred)
        mask_pred = mask_pred.resize((max_l, max_l)).crop(box)
    return mask_pred


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def get_padding(img):
    w, h = img.size
    max_size = max(w, h)
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    return pad_h, pad_w, max_size - pad_h - h, max_size - pad_w - w


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def predict_process():
    stride, slice_size, size, is_slice, mask_threshold, cpu, viz, no_save = 128, 256, 512, False, 0.1, False, False, False
    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'
    network = 'Ringed_Res_Unet'
    net_name = "resunet"
    file_names = os.listdir("data/test/imgs")
    model = 'result/best_epoch-00.bin'

    net = ResUnet(input_channel=3, output_channel=1)

    if not cpu:
        net.cuda()
        net.load_state_dict(torch.load(model))
        print("Load model:" + model)
    else:
        net.cpu()
        net.load_state_dict(torch.load(model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    for img_name in tqdm(file_names, total=len(file_names)):
        img_org = Image.open("data/test/imgs/" + img_name)
        img = data_transforms['img'](img_org)

        result = predict_img(net=net,
                             img_org=img_org,
                             full_img=img,
                             net_name=net_name,
                             slice=slice_size,
                             size=size,
                             stride=stride,
                             is_slice=is_slice,
                             out_threshold=mask_threshold,
                             use_gpu=not cpu)

        if viz:
            print("Visualizing results for image {}, close to continue ...".format(j))
            plot_img_and_mask(img, result)

        if not no_save:
            result.save('data/test/masks/' + net_name + "/" + img_name.split('.')[0] + '.png')

if __name__ == "__main__":
    train()