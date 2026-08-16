"""图像预处理（5.4.2）：ROI 裁剪 + 尺寸统一 + 对比度增强 + 归一化。

说明：YOLOv8 训练/推理内部自带 letterbox 缩放与归一化，
本模块提供显式的预处理步骤，供论文 5.4.2 的方法说明与
对照实验使用，可按需在数据加载前调用。
"""

import cv2
import numpy as np


def roi_crop(img, roi):
    """ROI 裁剪：roi = (x, y, w, h)。"""
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def resize_uniform(img, size=640):
    """尺寸统一：直接缩放到 size×size。"""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def clahe_enhance(img):
    """CLAHE 对比度增强（光照不均匀时有助于缺陷可见性）。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def normalize(img):
    """归一化到 [0, 1]。"""
    return img.astype(np.float32) / 255.0


def preprocess(img, size=640, roi=None, use_clahe=False, do_normalize=True):
    """完整预处理流水线。"""
    if roi is not None:
        img = roi_crop(img, roi)
    img = resize_uniform(img, size)
    if use_clahe:
        img = clahe_enhance(img)
    if do_normalize:
        img = normalize(img)
    return img
