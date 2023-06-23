import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import torch
from argparse import ArgumentParser
import  random
import os

class CFG:
    # training related
    parser = ArgumentParser(description='e-Lab Segmentation Script')
    arg = parser.add_argument
    arg('--bs', type=int, default=32, help='batch size')
    arg('--lr', type=float, default=5e-6, help='learning rate, default is 5e-4')
    arg('--lrd', type=float, default=1e-7, help='learning rate decay (in # samples)')
    arg('--wd', type=float, default=1e-6, help='L2 penalty on the weights, default is 2e-4')
    arg('--log_dir', type=str, default='reslut/logs/random/', help='logdir')

    # device related
    arg('--workers', type=int, default=4, help='# of cpu threads for data-loader')
    arg('--maxepoch', type=int, default=30, help='maximum number of training epochs')
    arg('--seed', type=int, default=42, help='seed value for random number generator')

    # data set related:
    arg('--datapath', type=str, default='data', help='dataset location')
    arg('--valid_id', type=str, default='random', choices=['1', '2', '3', 'random'],
        help='valid type')
    arg('--img_size', type=int, default=128, help='image height')

    # model related
    arg('--model', type=str, default='linknet', help='linknet')
    arg('--encoder_name', type=str, default='resnet3d',choices=['resnet3d','resnet3d_v2','r2plus1d_18','r3d_18','mc3_18','convnext3d','unet3d_down'] , help='type of encoder')
    arg('--decoder_name', type=str, default='cnn', choices=['cnn','cnn_v2','uper_head','unetplusplus','unet3d_up','cnn_res','cnn+uperhead'], help='type of decoder')
    arg('--mix_up', type=bool, default=False, help='type of decoder with mixed')
    arg('--checkpoint', type=str, default='result/resnet3d-seg-2d/resnet18/only_mask/random/Resnet3D-DIM-22-eval_loss-0.0915-dice_score-0.86-iou_score-0.78-30-epoch.pkl', help='model file path')
    arg('--pretrain', type=bool, default=False, help='load pretrain model')

    # Saving/Displaying Information
    arg('--resume', type=bool, default=False, help='Resume from previous checkpoint')

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pretrained = args.pretrain
    checkpoint = args.checkpoint
    model_name = args.model
    size = args.img_size
    classes = 1
    lr = args.lr
    min_lr = args.lrd
    epochs = args.maxepoch
    batch_size = args.bs
    min_loss = np.inf
    weight_decay = args.wd
    resume = args.resume
    num_workers = args.workers
    log_dir = args.log_dir
    seed = args.seed
    mix_up = args.mix_up
    use_amp = True

    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # # comp_dir_path = './'
    # comp_dir_path = '/kaggle/input/'
    # comp_folder_name = 'vesuvius-challenge-ink-detection'
    # # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    # comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
        # comp_dir_path = './'
    comp_dir_path = ''
    comp_folder_name = args.datapath
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    
    encoder_name = args.encoder_name # resnet3d-10（18）、r2plus1d_18、r3d_18、mc3_18、convnext3d、unet3d_down
    decoder_name = args.decoder_name # cnn、uper_head、unetplusplus、unet3d_up、cnn+uperhead

    # ============== pred target =============
    target_size = classes

    # ============== model cfg =============
    model_name = encoder_name + '-' + decoder_name

    in_idx = [i for i in range(21, 43)] # 21 43

    valid_id = args.valid_id if args.valid_id == 'random' else int(args.valid_id) # 1 2 3 random

    rate_valid = 0.2

    in_chans =  len(in_idx)# 22
    # ============== training cfg =============

    train_tile_size_1 = size
    train_stride_1 = train_tile_size_1 // 4

    train_tile_size_2 = size
    train_stride_2 = train_tile_size_2 // 2

    train_tile_size_3 = size
    train_stride_3= train_tile_size_3 // 4

    valid_tile_size = size
    valid_stride = valid_tile_size // 2

    train_batch_size = batch_size # 32
    valid_batch_size = batch_size
    use_amp = True

    inplanes = [64, 128, 256, 512]

    # ============== fixed =============
    max_grad_norm = 1000

    num_workers = 4

    threshhold = 0.5

    all_best_dice = 0
    all_best_loss = np.float('inf')

    shape_list = []
    test_shape_list = []

    val_mask = None
    val_label = None

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.2), ratio=(1.0, 1.0)),
        A.Resize(size, size),
        A.Rotate(limit=90, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
                A.GaussNoise(),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3), 
                        mask_fill_value=0, p=0.5),
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
