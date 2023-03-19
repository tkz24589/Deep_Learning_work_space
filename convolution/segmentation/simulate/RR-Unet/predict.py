import math

import numpy as np
from torch.utils.data import DataLoader

from model_seg import SwinTransformer, UPerHead
from model_seg.backbones.convnext import ConvNeXt
from model_seg.ops import resize
from unet.unet_model import *
from utils import *
from utils.data_vis import plot_img_and_mask
from tqdm import tqdm
from utils.utils import SquarePad
from utils.dataset import DTTDataSet
import cv2


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
        full_img = full_img.cuda()

    with torch.no_grad():
        if net_name == "RRUnet_class1":
            mask = net(full_img)
            mask_pred = (torch.sigmoid(mask).squeeze() > out_threshold).int().cpu().numpy()
        elif net_name == "RRUnet_class2":
            mask = net(full_img).squeeze()
            mask_0 = (torch.sigmoid(mask[:, 0, :, :]) > out_threshold).int()
            mask_1 = (torch.sigmoid(mask[:, 1, :, :]) > out_threshold).int()
            mask_pred = (((mask_0 + mask_1) / 2.) >= 0.5).int().cpu().numpy()
        elif net_name == "ConvNeXt":
            if is_slice:
                img_slices = []
                for i in range(full_img.shape[0]):
                    im = full_img[i].unsqueeze(0)
                    mas = net(im)
                    mas = resize(input=mas,
                                 size=full_img.shape[2:],
                                 mode='bilinear',
                                 align_corners=False)
                    mas = torch.sigmoid(mas)
                    img_slices.append(mas)
                mask_pred = torch.stack(img_slices).squeeze(1)[:, 1, :, :]
                mask_pred_normal = torch.stack(img_slices).squeeze(1)[:, 0, :, :]
                s = max(img_org.width, img_org.height) + slice
                mask_finnal = np.zeros((s, s))
                mask_finnal_normal = np.zeros((s, s))
                # 处理滑动窗口拼接
                mask_index = 0
                for i in range(0, img_org.width, stride):
                    for j in range(0, img_org.height, stride):
                        if mask_index > mask_pred.shape[0]:
                            break
                        mask_finnal[j:j + slice, i:i + slice] = mask_pred[mask_index].cpu().numpy()
                        mask_finnal_normal[j:j + slice, i:i + slice] = 1 - mask_pred_normal[mask_index].cpu().numpy()
                        mask_index += 1
                mask_pred = mask_finnal[:img_org.height, :img_org.width]
                mask_pred_normal = mask_finnal_normal[:img_org.height, :img_org.width]
                # mask_pred = mask_pred + mask_pred_normal
                mask_pred = mask_to_image(mask_pred)
            else:
                mask_pred = net(full_img)
                mask_pred = resize(input=mask_pred,
                             size=full_img.shape[2:],
                             mode='bilinear',
                             align_corners=False)
                mask_pred_forge = torch.sigmoid(mask_pred[:, 1, :, :]).squeeze()
                mask_pred_normal = mask_pred[:, 0, :, :]
                mask_pred = mask_to_image(mask_pred_forge.cpu().numpy())
        # 裁剪图片到原大小
        # padding = get_padding(img_org)
        # box = (padding[1], padding[0], padding[1] + img_org.width, padding[0] + img_org.height)
        # max_l = max(img_org.width, img_org.height)

        # mask_pred = mask_pred.resize((max_l, max_l)).crop(box)
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


if __name__ == "__main__":
    stride, slice_size, is_slice, mask_threshold, cpu, viz, no_save = 768, 768, False, 0.1, False, False, False
    n_class = 2
    batchsize = 2
    epochs = 50
    size = 768
    crop_size = 768
    classes = ["0", "1"]
    palette = [[0, 0, 0], [255, 255, 255]]
    gpu = True
    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'
    network = 'ConvNeXt'
    net_name = "ConvNeXt"
    # test_dataset = DTTDataSet(is_train=False,
    #                           data_root='data/test',
    #                           img_dir='imgs',
    #                           ann_dir='masks',
    #                           classes=classes,
    #                           palette=palette,
    #                           size=size,
    #                           crop_size=crop_size
    #                           )
    file_names = os.listdir("data/test/imgs")
    transform_img = transforms.Compose([
        # SquarePad(is_mask=False),
        # transforms.Resize(size),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4)
    for img_name in tqdm(file_names, total=len(file_names)):
        img_org = Image.open("data/test/imgs/" + img_name)
        img = transform_img(img_org)
        if is_slice:
            img_slices = []
            for i in range(0, img.width, stride):
                for j in range(0, img.height, stride):
                    img_slice = img.crop((i, j, i + slice_size, j + slice_size))
                    img_slices.append(normalize(transform(img_slice)))
            img = torch.stack(img_slices)
        else:
            img =normalize(transform(img).unsqueeze(0))
        model = 'result/ConvNeXt-[iou_score]-0.1446-[train_loss]-0.1316.pkl'

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

        if not cpu:
            net.cuda()
            net.load_state_dict(torch.load(model))
        else:
            net.cpu()
            net.load_state_dict(torch.load(model, map_location='cpu'))
            print("Using CPU version of the net, this may be very slow")

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
