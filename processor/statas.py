import glob
import warnings
from collections import Counter

import numpy as np
import rasterio
from tqdm import tqdm

from utils.misc import AverageMeter

warnings.filterwarnings("ignore")
import logging



def count_unique(image_re_path=None, image_list=None):
    if image_re_path is None:
        assert image_list is not None, "❌ 警告：image_path 和 image_list 不能同时为 None！请检查输入路径是否正确。"
    else:
        image_list = glob.glob(image_re_path)

    label_counter = Counter()
    total_pixels = 0

    for filename in tqdm(image_list):
        with rasterio.open(filename) as src:
            img = src.read()
            unique, counts = np.unique(img, return_counts=True)
            label_counter.update(dict(zip(unique, counts)))
            total_pixels += img.size  # 高 × 宽，总像素数

    print("✅ 类别像素统计：")
    for label, count in sorted(label_counter.items()):
        ratio = count / total_pixels
        print(f"类别 {label}: {count} 像素，占比 {ratio:.4%}")
    pass


def count_rgb_unique(image_re_path=None, image_list=None):
    if image_re_path is None:
        assert image_list is not None, "❌ 警告：image_path 不能为 None！请检查输入路径是否正确。"
    else:
        image_list = glob.glob(image_re_path)

    label_counter = Counter()
    total_pixels = 0

    for filename in tqdm(image_list):
        with rasterio.open(filename) as src:
            img = src.read()
            img_flat = img.reshape(3, -1).transpose(1, 0)
            unique, counts = np.unique(img_flat[::100], return_counts=True, axis=0)
            label_counter.update({tuple(int(c) for c in rgb): int(count) for rgb, count in zip(unique, counts)})
            total_pixels += counts.sum()

    print("✅ 类别像素统计：")
    for label, count in sorted(label_counter.items()):
        ratio = count / total_pixels
        print(f"类别 {label}: {count} 像素，占比 {ratio:.4%}")
    pass


def stata_mean_std(image_re_path=None, image_list=None):
    if image_re_path is None:
        assert image_list is not None, "❌ 警告：image_path 不能为 None！请检查输入路径是否正确。"
    else:
        image_list = glob.glob(image_re_path)

    mean_record = AverageMeter()
    std_record = AverageMeter()

    for filename in tqdm(image_list):
        with rasterio.open(filename) as src:
            img = src.read()
            c = img.shape[0]
            img_flat = img.reshape(c, -1).transpose(1, 0)
            mean = np.nanmean(img_flat[::10], axis=0)
            std = np.nanstd(img_flat[::10], axis=0)
            mean_record.update(mean)
            std_record.update(std)
    print(f"mean {mean_record.avg}；std {std_record.avg}")
    pass
