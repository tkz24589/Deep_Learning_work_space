import torch
import torch.backends.cudnn as cudnn
from torch import optim

from eval import eval_net
from unet.unet_model import *
from utils import *

import matplotlib.pyplot as plt
import time

from utils.dice_score import dice_loss


def train_net(net,
              epochs=1,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=1,
              dataset=None):
    # training images are square
    ids = split_ids(get_ids(dir_img))
    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs,
               batch_size,
               lr,
               len(iddataset['train']),
               len(iddataset['val']),
               str(save_cp),
               str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.Adam(net.parameters(),
                          lr=lr,
                        weight_decay=0)

    criterion = nn.BCELoss()

    Train_loss = []
    Valida_dice = []
    EPOCH = []

    for epoch in range(epochs):
        net.train()

        start_epoch = time.time()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale, dataset)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale, dataset)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            start_batch = time.time()
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b]).astype(np.float32)

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            optimizer.zero_grad()

            masks_pred = net(imgs)
            masks_probs = torch.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)
            true_masks_flat = true_masks.view(-1)
            # loss = criterion(masks_probs_flat, true_masks_flat)
            loss = dice_loss(masks_probs_flat, true_masks_flat, multiclass=False)
            # loss = torch.mul(loss, 0.6)
            # loss_dice = torch.mul(loss_dice, 0.4)
            # loss = loss_dice + loss
            print('epoch: {:.4f} --- loss: {:.4f}, time: {:.3f}s'.format(i * batch_size / N_train, loss, time.time() - start_batch))

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {:.4f}'.format(epoch_loss / i))

        # validate the performance of the model
        net.eval()

        val_dice = eval_net(net, val, gpu)
        print('Validation Dice Coeff: {:.4f}'.format(val_dice))

        Train_loss.append(epoch_loss / i)
        Valida_dice.append(val_dice)
        EPOCH.append(epoch)
        #
        fig = plt.figure()

        plt.title('Training Process')
        plt.xlabel('epoch')
        plt.ylabel('value')
        l1, = plt.plot(EPOCH, Train_loss, c='red')
        l2, = plt.plot(EPOCH, Valida_dice, c='blue')

        plt.legend(handles=[l1, l2], labels=['Tra_loss', 'Val_dice'], loc='best')
        plt.savefig(dir_logs + 'Training Process for lr-{}.png'.format(lr), dpi=600)

        torch.save(net.state_dict(),
                   dir_logs + '{}-[val_dice]-{:.4f}-[train_loss]-{:.4f}.pkl'.format(dataset, val_dice, epoch_loss / i))
        print('Spend time: {:.3f}s'.format(time.time() - start_epoch))
        print()


if __name__ == '__main__':
    epochs, batchsize, scale, gpu = 50, 1, 1, True
    lr = 1e-3
    ft = True
    dataset = 'CASIA'

    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'
    model = 'Ringed_Res_Unet'

    dir_img = 'data/JPEGImages/'.format(dataset)
    dir_mask = 'data/SegmentationClassPNG/'.format(dataset)
    dir_logs = 'result/logs/{}/{}/'.format(dataset, model)

    if model == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif model == 'Res_Unet':
        net = Res_Unet(n_channels=3, n_classes=1)
    elif model == 'Ringed_Res_Unet':
        net = Ringed_Res_Unet(n_channels=3, n_classes=1)

    if ft:
        fine_tuning_model = 'result/logs/CASIA/Ringed_Res_Unet/PEE-CASIA-[val_dice]-0.9900-[train_loss]-0.0687.pkl'.format(dataset, model)
        net.load_state_dict(torch.load(fine_tuning_model))
        print('Model loaded from {}'.format(fine_tuning_model))

    if gpu:
        net.cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory

    train_net(net=net,
              epochs=epochs,
              batch_size=batchsize,
              lr=lr,
              gpu=gpu,
              img_scale=scale,
              dataset=dataset)
# import argparse
# import logging
# import sys
# from pathlib import Path
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import wandb
# from torch import optim
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
#
# from utils.data_loading import BasicDataset, CarvanaDataset
# from utils.dice_score import dice_loss
# from evaluate import evaluate
# from unet.unet_model import *
#
# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')
#
#
# def train_net(net,
#               device,
#               epochs: int = 5,
#               batch_size: int = 1,
#               learning_rate: float = 1e-5,
#               val_percent: float = 0.1,
#               save_checkpoint: bool = True,
#               img_scale: float = 0.5,
#               amp: bool = False):
#     # 1. Create dataset
#     try:
#         dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
#     except (AssertionError, RuntimeError):
#         dataset = BasicDataset(dir_img, dir_mask, img_scale)
#
#     # 2. Split into train / validation partitions
#     n_val = int(len(dataset) * val_percent)
#     n_train = len(dataset) - n_val
#     train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
#
#     # 3. Create data loaders
#     loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
#     train_loader = DataLoader(train_set, shuffle=True, **loader_args)
#     val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
#
#     # (Initialize logging)
#     experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
#     experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
#                                   val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
#                                   amp=amp))
#
#     logging.info(f'''Starting training:
#         Epochs:          {epochs}
#         Batch size:      {batch_size}
#         Learning rate:   {learning_rate}
#         Training size:   {n_train}
#         Validation size: {n_val}
#         Checkpoints:     {save_checkpoint}
#         Device:          {device.type}
#         Images scaling:  {img_scale}
#         Mixed Precision: {amp}
#     ''')
#
#     # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
#     optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
#     grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
#     criterion = nn.BCEWithLogitsLoss()
#     global_step = 0
#
#     # 5. Begin training
#     for epoch in range(epochs):
#         net.train()
#         epoch_loss = 0
#         with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
#             for batch in train_loader:
#                 images = batch['image']
#                 true_masks = batch['mask']
#
#                 assert images.shape[1] == net.n_channels, \
#                     f'Network has been defined with {net.n_channels} input channels, ' \
#                     f'but loaded images have {images.shape[1]} channels. Please check that ' \
#                     'the images are loaded correctly.'
#
#                 images = images.to(device=device, dtype=torch.float32)
#                 true_masks = true_masks.to(device=device, dtype=torch.float16)
#
#                 with torch.cuda.amp.autocast(enabled=amp):
#                     masks_pred = net(images)
#                     masks_probs_flat = masks_pred.view(-1)
#                     true_masks_flat = true_masks.view(-1)
#                     loss = criterion(masks_probs_flat, true_masks_flat) \
#                            + dice_loss(torch.sigmoid(masks_pred).float(),
#                                        true_masks,
#                                        multiclass=False)
#
#                 optimizer.zero_grad(set_to_none=True)
#                 grad_scaler.scale(loss).backward()
#                 grad_scaler.step(optimizer)
#                 grad_scaler.update()
#
#                 pbar.update(images.shape[0])
#                 global_step += 1
#                 epoch_loss += loss.item()
#                 experiment.log({
#                     'train loss': loss.item(),
#                     'step': global_step,
#                     'epoch': epoch
#                 })
#                 pbar.set_postfix(**{'loss (batch)': loss.item()})
#
#                 # Evaluation round
#                 division_step = (n_train // (10 * batch_size))
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                         histograms = {}
#                         for tag, value in net.named_parameters():
#                             tag = tag.replace('/', '.')
#                             histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                             histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
#
#                         val_score = evaluate(net, val_loader, device)
#                         scheduler.step(val_score)
#
#                         logging.info('Validation Dice score: {}'.format(val_score))
#                         experiment.log({
#                             'learning rate': optimizer.param_groups[0]['lr'],
#                             'validation Dice': val_score,
#                             'images': wandb.Image(images[0].cpu()),
#                             'masks': {
#                                 'true': wandb.Image(true_masks[0].float().cpu()),
#                                 'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
#                             },
#                             'step': global_step,
#                             'epoch': epoch,
#                             **histograms
#                         })
#
#         if save_checkpoint:
#             Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
#             torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
#             logging.info(f'Checkpoint {epoch + 1} saved!')
#
#
# def get_args():
#     parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
#     parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
#     parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=6, help='Batch size')
#     parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
#                         help='Learning rate', dest='lr')
#     parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
#     parser.add_argument('--scale', '-imgs', type=float, default=0.5, help='Downscaling factor of the images')
#     parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
#                         help='Percent of the data that is used as validation (0-100)')
#     parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     args = get_args()
#
#     logging.basicConfig(level=logging.INFO, format='%(levelname)imgs: %(message)imgs')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Using device {device}')
#
#     # Change here to adapt to your data
#     # n_channels=3 for RGB images
#     # n_classes is the number of probabilities you want to get per pixel
#     net = Ringed_Res_Unet(n_channels=3, n_classes=1)
#
#     logging.info(f'Network:\n'
#                  f'\t{net.n_channels} input channels\n'
#                  f'\t{net.n_classes} output channels (classes)\n')
#                  # f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
#
#     if args.load:
#         net.load_state_dict(torch.load(args.load, map_location=device))
#         logging.info(f'Model loaded from {args.load}')
#
#     net.to(device=device)
#     try:
#         train_net(net=net,
#                   epochs=args.epochs,
#                   batch_size=args.batch_size,
#                   learning_rate=args.lr,
#                   device=device,
#                   img_scale=args.scale,
#                   val_percent=args.val / 100,
#                   amp=args.amp)
#     except KeyboardInterrupt:
#         torch.save(net.state_dict(), 'INTERRUPTED.pth')
#         logging.info('Saved interrupt')
#         sys.exit(0)
