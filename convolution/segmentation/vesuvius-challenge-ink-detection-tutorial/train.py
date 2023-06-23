import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import numpy as np
from torch import optim
from tqdm import tqdm
import gc
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from model.ink_model import Ink3DModel, Ink3DUnet
from config.config import CFG
from utils.dataset import Ink_Detection_Dataset, get_transforms
from utils.scheduler import get_scheduler, scheduler_step
from utils.score import dice_coef
from utils.utils import get_random_train_valid_dataset, get_train_valid_dataset
from utils.loss import multiLoss


def train_step(train_loader, model, criterion, optimizer, writer, device, epoch):
    model.train()
    epoch_loss = 0
    scaler = GradScaler(enabled=CFG.use_amp)
    bar = tqdm(enumerate(train_loader), total=len(train_loader)) 
    for step, (image, label) in bar:
        optimizer.zero_grad()
        outputs = model(image.to(device))
        loss = criterion.loss(outputs, label.to(device))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{loss.item():0.4f}', epoch=epoch ,gpu_mem=f'{mem:0.2f} GB', lr=f'{optimizer.state_dict()["param_groups"][0]["lr"]:0.2e}')
        epoch_loss += loss.item()
    writer.add_scalar('Train/Loss', epoch_loss / len(train_loader), epoch)
    return epoch_loss / len(train_loader)


def valid_step(valid_loader, model, valid_xyxys, valid_mask , criterion, device, writer, epoch):
    model.eval()
    if CFG.valid_id != 'random':
        mask_pred = np.zeros(valid_mask.shape)
        mask_count = (1 - valid_mask).astype(np.float64)
        valid_mask_gt = np.zeros(valid_mask.shape)

    epoch_loss = 0
    best_th = 0
    best_dice = 0
    dice_scores = {}
    for th in np.arange(1, 8, 0.5) / 10:
        dice_scores[th] = []

    bar = tqdm(enumerate(valid_loader), total=len(valid_loader)) 
    for step, (image, label) in bar:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_pred = model(image)
            loss = criterion.loss(y_pred, label)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{loss.item():0.4f}', epoch=epoch ,gpu_mem=f'{mem:0.2f} GB')
        # make whole mask
        if CFG.valid_id != 'random':
            y_pred = torch.sigmoid(y_pred).to('cpu').numpy()
            label = label.to('cpu').numpy()
            start_idx = step*CFG.valid_batch_size
            end_idx = start_idx + CFG.valid_batch_size
            for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
                mask_pred[y1:y2, x1:x2] += y_pred[i].squeeze(0)
                valid_mask_gt[y1:y2, x1:x2] = label[i].squeeze(0)
                mask_count[y1:y2, x1:x2] += np.ones((CFG.valid_tile_size, CFG.valid_tile_size))
        else:
            y_pred = torch.sigmoid(y_pred)
            for th in np.arange(1, 8, 0.5) / 10:
                dice_score = dice_coef(label, y_pred, thr=th).item()
                dice_scores[th].append(dice_score)
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(valid_loader)
    writer.add_scalar('Valid/Loss', avg_loss, epoch)
    if CFG.valid_id != 'random':
        print(f'mask_count_min: {mask_count.min()}')
        mask_pred /= mask_count
        mask_pred *= valid_mask
        has_nan = np.isnan(mask_pred).any()
        print(has_nan)
        if CFG.valid_id == 2:
            # 防止内存溢出if  y2 <4800 or y2 > 4800 + 4096 + 2048 or x2 > 640+ 4096 +2048 or x2 < 640:
            valid_mask_gt = valid_mask_gt[4800:4800+4096+2048, 640:640+4096+2048]
            mask_pred = mask_pred[4800:4800+4096+2048, 640:640+4096+2048]
            valid_mask = valid_mask[4800:4800+4096+2048, 640:640+4096+2048]
        for th in np.arange(1, 8, 0.5) / 10:
            dice_score = dice_coef(torch.from_numpy(valid_mask_gt).to(CFG.device), torch.from_numpy(mask_pred).to(CFG.device), thr=th).item()
            dice_scores[th].append(dice_score)
        for th in np.arange(1, 8, 0.5) / 10:
            dice_score = sum(dice_scores[th]) / len(dice_scores[th])
            if dice_score > best_dice:
                best_dice = dice_score
                best_th = th
        mask_pred = (mask_pred >= best_th).astype(int)
        cv2.imwrite(CFG.log_dir + f'{epoch}.png', mask_pred * 255)
        cv2.imwrite(CFG.log_dir + 'gt.png', valid_mask_gt * 255)
    else:
        print(dice_scores.keys())
        print('slice dice:')
        for th in np.arange(1, 8, 0.5) / 10:
            dice_score = sum(dice_scores[th]) / len(dice_scores[th])
            if dice_score > best_dice:
                best_dice = dice_score
                best_th = th
    print(best_dice, best_th)
    if CFG.all_best_dice < best_dice:
        print('best_th={:2f}' .format(best_th),"score up: {:2f}->{:2f}".format(CFG.all_best_dice, best_dice))       
        CFG.all_best_dice = best_dice
    # torch.save(model.state_dict(), 'result/' +  '{}-DIM-{}-[eval_loss]-{:.4f}-[dice_score]-{:.2f}-'.format(CFG.model_name, CFG.in_chans , avg_loss, best_dice) + str(epoch) + '-epoch.pkl')  
    # 保存checkpoint
    # torch.save({'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             },
    #            CFG.log_dir +  '{}-DIM-{}-eval_loss-{:.4f}-dice_score-{:.2f}-best_th-{:.2f}-valid-{}-'.format(CFG.model_name, CFG.in_chans , avg_loss, best_dice, best_th ,CFG.valid_id) + str(epoch) + '-epoch.pth')
    torch.save(model.state_dict(),
               CFG.log_dir + '{}-DIM-{}-eval_loss-{:.4f}-dice_score-{:.2f}-best_th-{:.2f}-valid-{}-'.format(CFG.model_name, CFG.in_chans , avg_loss, best_dice, best_th ,CFG.valid_id) + str(epoch) + '-epoch.pth')
    writer.add_scalar('Valid/Dice', best_dice, epoch)
    
    return avg_loss

if __name__ == '__main__':
    if CFG.decoder_name == 'unetplusplus':
        model = Ink3DUnet(encoder_name=CFG.encoder_name, 
                        decoder_name=CFG.decoder_name,
                        classes=CFG.target_size ,
                        decoder_attention_type='scse', 
                        encoder_depth=4, 
                        decoder_channels=CFG.inplanes[::-1])
    else:
        model = Ink3DModel(encoder_name=CFG.encoder_name, decoder_name=CFG.decoder_name, classes=CFG.target_size)
    # print(model)
    # i = torch.ones((2, 1, 22, 224, 224))
    # print(model(i).shape)
    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(CFG.device)

    criterion = multiLoss(1.0, 0)
    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG.lr,
                            betas=(0.9, 0.999),
                            weight_decay=CFG.weight_decay
                            )
    # optimizer = optim.SGD(model.parameters(),
    #                     lr=CFG.lr,
    #                     weight_decay=CFG.weight_decay
    #                     )
    scheduler = get_scheduler(CFG, optimizer)
    writer = SummaryWriter(CFG.log_dir)
    start_epoch = 0

    if CFG.pretrained:
        checkpoint = torch.load(CFG.checkpoint)
        try:
            model.load_state_dict(checkpoint)
            print('Checkpoint loaded')
        except:
            models_dict = model.state_dict()
            for model_part in models_dict:
                if model_part in checkpoint:
                    models_dict[model_part] = checkpoint[model_part]
            model.load_state_dict(models_dict)
            print('Checkpoint part loaded')

    if CFG.valid_id == "random":
        train_images, train_masks, valid_images, valid_masks, _ = get_random_train_valid_dataset(CFG.rate_valid)
    else:
        train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()
        valid_xyxys = np.stack(valid_xyxys)

    if CFG.valid_id != 'random':
        valid_xyxys = np.stack(valid_xyxys)
    else:
        valid_xyxys = None

    train_dataset = Ink_Detection_Dataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
    valid_dataset = Ink_Detection_Dataset(
        valid_images, CFG, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

    train_loader = data.DataLoader(train_dataset,
                            batch_size=CFG.train_batch_size,
                            shuffle=True,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                            )
    valid_loader = data.DataLoader(valid_dataset,
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    # 是否使用整张图片进行dice计算
    if CFG.valid_id != 'random':
        fragment_id = CFG.valid_id
        valid_mask = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/mask.png", 0)
        valid_mask = valid_mask.astype('float32') / 255.
        pad0 = (CFG.valid_tile_size - valid_mask.shape[0] % CFG.valid_tile_size)
        pad1 = (CFG.valid_tile_size - valid_mask.shape[1] % CFG.valid_tile_size)
        valid_mask = np.pad(valid_mask, [(0, pad0), (0, pad1)], constant_values=0)
    else:
        valid_mask = None


    for i in range(start_epoch, CFG.epochs):
        print('train:')
        train_step(train_loader, model, criterion, optimizer, writer, CFG.device, i + 1)
        print('val:')
        val_loss = valid_step(valid_loader, model, valid_xyxys, valid_mask, criterion, CFG.device, writer,  i + 1)
        scheduler_step(scheduler, val_loss, i + 1)
        gc.collect()
