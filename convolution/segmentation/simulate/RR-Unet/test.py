# import cv2
# import numpy as np
# import os

# # # 读取图像
# # img_list = os.listdir("data/train/untampered/")
# # # mask_list = os.listdir("data/train/tampered/masks")
# # img_root = "data/train/untampered/"
# # mask_root = "data/train/tampered/masks/"
# # new_img_root = "data/train/tampered/imgs/"
# # # new_mask_root = "data/train/tampered/new_masks/"
# # for i in range(len(img_list)):
# #     img_path = img_root + img_list[i]
# #     img = cv2.imread(img_path)
# #     img_mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# #     # 将灰度图像进行二值化处理
# #     mask = np.zeros_like(img_mask)
# #     cv2.imwrite(mask_root + img_list[i].split('.')[0] + '_ut.png', mask)
# #     cv2.imwrite(new_img_root + img_list[i].split('.')[0] + '_ut.jpg', img)
# # 读取图像
import os
mask_root = "data/train/tampered/masks/"
img_root = "data/train/tampered/imgs/"
new_mask_root = "data/train/tampered/new_masks/"
new_img_root = "data/train/tampered/new_imgs/"
mask_list_all = os.listdir(mask_root)
img_list_all = os.listdir(img_root)
mask_list = []
img_list = []
for i in mask_list_all:
    if "_ut.png" in i:
        os.remove(mask_root + i)
for i in img_list_all:
    if "_ut.jpg" in i:
        os.remove(img_root + i)
        
# for i in mask_list:
#     mask_path = mask_root + i
#     img_path = img_root + i.split('.')[0] + '.jpg'

#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     mask_flag = list((mask.flatten() > 127).astype(int))
#     img = cv2.imread(img_path)
#     if mask_flag.count(1) > 1000:
#         cv2.imwrite(new_mask_root + i, mask)
#         cv2.imwrite(new_img_root + i.split('.')[0] + '.jpg', img)
# for i in mask_list:
#     mask_path = mask_root + i
#     img_path = img_root + i.split('.')[0] + '.jpg'

#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     mask_flag = list((mask.flatten() > 127).astype(int))
#     rectangle_mask = np.zeros_like(mask)
#     for i in range(len(mask_flag)):
#         for j in range(len(mask_flag[0])):
#             if mask_flag[i][j] == 1:
#                 # 从修改目标的左上角开始
#                 x, y = i, j
#                 while mask_flag[x][y] == 0 or 
#     img = cv2.imread(img_path)
#     cv2.imwrite(new_mask_root + i, mask)
#     cv2.imwrite(new_img_root + i.split('.')[0] + '.jpg', img)
import cv2
import torch
# img = cv2.imread("data/train/tampered/masks/0007.png", 0)
# img = torch.Tensor(list(img.flatten())) > 127
# img = img.float().numpy().tolist()
# print(str(255) + ":" + str(img.count(True)))
file = os.listdir('data/train/tampered/img')
img_root = 'data/train/tampered/img/'
mask_root  = 'data/train/tampered/mask/'
j = 4001
for i in file:
    img = cv2.imread(img_root + i[:-4] + '.jpg')
    mask = cv2.imread(mask_root + i[:-4] + '.png', 0)
    cv2.imwrite(img_root + str(j) + '.jpg', img)
    cv2.imwrite(mask_root + str(j) + '.png', mask)
    j += 1





    