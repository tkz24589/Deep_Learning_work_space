import numpy as np
from glob import glob
import cv2
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def score_cls(submission_path, labels):
    """
    submission_path: 'competition1/815903/815903_2023-02-13 18:48:29.csv'
    labels: labels = pd.read_csv('competition1/labels.txt',
                         delim_whitespace=True, header=None).to_numpy()
    赛道1评分
    """
    submission = pd.read_csv(
        submission_path, delim_whitespace=True, header=None).to_numpy()
    tampers = labels[labels[:, 1] == 1]
    untampers = labels[labels[:, 1] == 0]
    pred_tampers = submission[np.in1d(submission[:, 0], tampers[:, 0])]
    pred_untampers = submission[np.in1d(submission[:, 0], untampers[:, 0])]

    thres = np.percentile(pred_untampers[:, 1], np.arange(90, 100, 1))
    recall = np.mean(np.greater(pred_tampers[:, 1][:, np.newaxis], thres).mean(axis=0))
    return recall * 100


def get_threshold(submission_folder, untampers: list):
    preds = [np.max(cv2.imread(os.path.join(submission_folder, 'submission', path), cv2.IMREAD_GRAYSCALE))
            for path in untampers[:, 0]]
    return np.percentile(preds, np.arange(91, 100, 1))


def iou(pred, mask, thres):
    mask = np.clip(mask, 0, 1)
    iou_m = []

    first_threshold = int(thres[0])
    num_thresholds = 256 - first_threshold
    iou_map = np.zeros(num_thresholds)
    for i in range(num_thresholds):
        tmp = np.zeros_like(pred)
        thre = i + first_threshold
        tmp[pred > thre] = 1
        iou_value = np.count_nonzero(np.logical_and(
            mask, tmp)) / np.count_nonzero(np.logical_or(mask, tmp))
        iou_map[i] = iou_value 
    for item in thres:
        iou_m.append(np.max(iou_map[int(item)-first_threshold:]))
    return np.sum(iou_m)



def subprocess(submission_folder, thres, item):
    imgpath = os.path.join(submission_folder, 'submission', item)
    maskpath = os.path.join('competition2/ground_truth', item)
    pred = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    iou_score = iou(pred, mask, thres)
    return iou_score


def score_seg(submission_folder, labels):
    """
    submission_folder: 'competition2/815914/2023-02-14 14:06:41'
    labels: labels = pd.read_csv('competition2/ground_truth/labels.txt',
                         delim_whitespace=True, header=None).to_numpy()
    赛道2评分
    """
    tampers = labels[labels[:, 1] == 1]
    untampers = labels[labels[:, 1] == 0]
    thres = get_threshold(submission_folder, untampers)
    
    num_tampers = tampers.shape[0]
    scores = np.empty(num_tampers)
    with ProcessPoolExecutor() as ex:
        future_scores = [ex.submit(subprocess, submission_folder, thres, item) for item in tampers[:, 0]]
        for i, future in enumerate(as_completed(future_scores)):
            scores[i] = future.result()

    return np.mean(scores)
