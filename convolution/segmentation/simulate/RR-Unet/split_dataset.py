import numpy as np
import os
from utils.utils import split_train_val
import shutil

if __name__ == "__main__":
    img_root = "data/imgs_plus/"
    mask_root = "data/masks_plus/"

    train_root = "data/train/"
    val_root = "data/val/"

    file_list = os.listdir(img_root)

    ids = list(maskname[:-4] for maskname in file_list)

    split_data = split_train_val(ids, val_percent=0.1)

    for file in file_list:
        if file[:-4] in split_data['train']:
            shutil.copy(img_root + file, train_root + 'imgs/' + file)
            shutil.copy(mask_root + file[:-4] + '.png', train_root + 'masks/' + file[:-4] + '.png')
        else:
            shutil.copy(img_root + file, val_root + 'imgs/' + file)
            shutil.copy(mask_root + file[:-4] + '.png', val_root + 'masks/' + file[:-4] + '.png')


