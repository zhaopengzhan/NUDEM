import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import rasterio
import os
from einops import rearrange
import torch.nn.functional as F
import glob
from tqdm import tqdm
from PIL import Image
import cv2
import geopandas as gpd

from hydrology.modeling_regressor import AutoRegressor
from hydrology.water_area_curve import HypsometricCurve

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def main():
    dem_gt_path = r'F:\BaiduNetdiskDownload\cdm_contour\小柴旦湖_utm_tmp.tif'
    # dem_re_path = r'F:\BaiduNetdiskDownload\cdm_contour\AUNDEM.tif'
    dem_re_path = r'F:\zpz\Projects6\临时 处理DEM数据确保画出图\裁剪代码\data\epoch_9600_real.tif'

    with rasterio.open(dem_gt_path) as gt_ds:
        gt = gt_ds.read(1, masked=True)
        T = gt_ds.transform
        pixel_area = abs(T.a * T.e)

    hc_gt = HypsometricCurve(dem=gt, pixel_area=pixel_area)
    hc_re = HypsometricCurve(dem_path=dem_re_path)

    vmin, vmax = hc_gt.S[1200], hc_gt.S[-100]
    N_sample = 50
    areas = np.linspace(vmin, vmax, num=N_sample)

    heights_gt = hc_gt.get_height_from_area(areas)
    heights_re = hc_re.get_height_from_area(areas)

    model = AutoRegressor(areas, heights_gt, 'linear')

    model.show()
    pass


if __name__ == '__main__':
    main()

    pass
