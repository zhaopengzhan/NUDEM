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
from hydrology.water_area_curve import HypsometricCurve
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def main():
    path = r'F:\BaiduNetdiskDownload\cdm_contour\小柴旦湖_utm_tmp.tif'

    with rasterio.open(path) as src:
        dem = src.read(1, masked=True)
        T = src.transform
        pixel_area_m2 = abs(T.a * T.e)

    curve = HypsometricCurve(dem=dem, pixel_area=pixel_area_m2)

    curve.show()


if __name__ == '__main__':
    main()

    pass
