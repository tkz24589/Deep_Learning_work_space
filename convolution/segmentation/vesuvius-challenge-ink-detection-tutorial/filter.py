import cv2
import numpy as np
output_list = []
THRESHOLD = 0.55
for index in range(2):
    mask = cv2.imread(str(index + 1) + ".png", 0) # 读取灰度图像
    _, mask_binary = cv2.threshold(mask, int(255 * THRESHOLD), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)) # 创建一个3x3的椭圆形结构元素
    mask_opened = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_opened)
    min_area = 100
    max_num_regions = 100
    # 找到面积最大的区域
    max_areas = []
    max_labels = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if len(max_areas) < max_num_regions:
            max_areas.append(area)
            max_labels.append(i)
        else:
            min_idx = max_areas.index(min(max_areas))
            if area > max_areas[min_idx]:
                max_areas[min_idx] = area
                max_labels[min_idx] = i

    # 找到宽度和高度接近的区域
    max_width_height_ratio = 2.0
    selected_labels = []
    target_mask = np.zeros_like(labels)
    for i in range(len(max_labels)):
        label = max_labels[i]
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if abs(width - height) / max(width, height) > max_width_height_ratio:
            continue
        selected_labels.append(label)

    cv2.imwrite("mask_labels" + str(index + 1) + ".jpg", labels)
    # 提取目标区域
    masks_target = []
    for label in selected_labels:
        mask_target = (labels == label).astype("uint8") * 255
        target_mask += mask_target

    cv2.imwrite("mask_target_org" + str(index + 1) + ".jpg", target_mask)
    # 进行膨胀和腐蚀操作
    kernel = np.ones((50,50), np.uint8)
    dilated_mask = cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=1)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)

    # 将 mask 中的缺陷部分替换为原始图像中对应的部分
    result = np.copy(target_mask)
    result[eroded_mask != 0] = 255  # 将腐蚀后的 mask 中值为 0 的部分替换为白色
    # 提取目标区域
    cv2.imwrite("mask_target" + str(index + 1) + ".jpg", result)
    output_list.append(result / 255.)

def rle(output):
    flat_img = np.where(output.flatten() > 0.55, 1, 0).astype(np.uint8)
    print(max(flat_img))
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    return " ".join(map(str, sum(zip(starts_ix, lengths), ())))

rle_list = []
for output in output_list:
    rle_sample = rle(output)
    rle_list.append(rle_sample)
print("Id,Predicted\na," + rle_list[0] + "\nb," + rle_list[1], file=open('submission.csv', 'w'))