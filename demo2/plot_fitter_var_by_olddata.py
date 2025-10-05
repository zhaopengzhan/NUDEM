import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import mean_squared_error, r2_score

from hydrology.modeling_regressor import AutoRegressor
from hydrology.water_area_curve import HypsometricCurve

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

data = [
    [1999, 62.8803,     3120.387832,  53935318.5],
    [2000, 57.65868353, 3120.007596, -22909425.39],
    [2001, 48.57285671, 3119.346082, -35093950.42],
    [2002, 90.32025882, 3122.386757, 207909644.8],
    [2003, 79.69428,    3121.612532, -65771877.48],
    [2004, 69.84878965, 3120.895346, -53586333.93],
    [2005, 77.63896989, 3121.462801,  41826887.19],
    [2006, 77.89016645, 3121.481101,   1423050.165],
    [2007, 78.8580405,  3121.551611,   5526113.233],
    [2008, 86.79802609, 3122.130101,  47896844.73],
    [2009, 82.53309267, 3121.819354, -26306797.94],
    [2010, 97.82993691, 3122.93404,  100403363.3],
    [2011, 92.3768977,  3122.536629, -37789914.36],
    [2012, 100.6743612, 3123.141358,  58353818.94],
    [2013, 94.41273811, 3122.684993, -44507816.5],
    [2014, 87.29353607, 3122.166206, -47121344.41],
    [2015, 88.95511119, 3122.28728,   10669382.39],
    [2016, 98.44758727, 3122.979056,  64792680.56],
    [2017, 102.6537219, 3123.285633,  30824233.27],
    # [2018, 124.1065833, 3124.849757, 177075469.8],
    # [2019, 130.3563904, 3125.305577,  57988764.58],
    # [2020, 128.5254302, 3125.172032, -17286041.58],
    # [2021, 120.6363294, 3124.596688, -71664776.65],
    # [2022, 127.3047198, 3125.083,     60280985.94],
    # [2023, 123.3863469, 3124.797232, -35818231.74],
    # [2024, 126.2204468, 3125.003921,  25794904.79],
]

def compare_gt_vs_rec():
    dem_gt_path = r'F:\BaiduNetdiskDownload\cdm_contour\小柴旦湖_utm_tmp.tif'
    # dem_re_path = r'F:\BaiduNetdiskDownload\cdm_contour\AUNDEM.tif'
    dem_re_path = r'F:\zpz\Projects6\临时 处理DEM数据确保画出图\裁剪代码\data\epoch_9600_real.tif'

    with rasterio.open(dem_gt_path) as gt_ds:
        gt = gt_ds.read(1, masked=True)
        T = gt_ds.transform
        pixel_area = abs(T.a * T.e)

    hc_gt = HypsometricCurve(dem=gt, pixel_area=pixel_area)
    hc_re = HypsometricCurve(dem_path=dem_re_path)

    # TODO: 只修改这个地方
    df = pd.DataFrame(data, columns=["Year", "Area", "Height", "Var"])
    areas = df["Area"].values * 1e6

    heights_gt = hc_gt.get_height_from_area(areas)

    # TODO: 只改了这个地方
    model = AutoRegressor(areas[::3], heights_gt[::3], 'power')
    heights_re = model.get_y_from_x(areas)
    model.show()

    heights_gt = np.diff(heights_gt)
    heights_re = np.diff(heights_re)
    areas = (np.array(areas[:-1]) + np.array(areas[1:]))

    mse = mean_squared_error(heights_gt, heights_re)
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(heights_gt, heights_re)
    r2 = r2_score(heights_gt, heights_re)
    relative_bias = (np.sum(np.array(heights_re) - np.array(heights_gt)) /
                     np.sum(heights_gt)) * 100

    from matplotlib import pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["axes.unicode_minus"] = False

    # plot
    fig_width, fig_height = (11, 8)
    font_size = round(fig_height * 2.25)
    plt.rcParams.update({'font.size': font_size})

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plt.scatter(heights_gt, heights_re, c='r', alpha=0.7, label="Observed vs Predicted")
    # 画 1:1 参考线
    min_val = min(min(heights_gt), min(heights_re))
    max_val = max(max(heights_gt), max(heights_re))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="1:1 reference")

    plt.xlabel("Observed storage change (m$^2$)")
    plt.ylabel("Predicted storage change (m$^2$)")

    textstr = f"R2 = {r2:.3f}\nMSE = {mse:.3f}\nbias = {relative_bias:.2f}%"
    plt.text(0.8, 0.16, textstr, transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # plt.xlim(0, 0.03)
    # plt.ylim(0, 0.03)
    plt.title("Hypsometric Curve of Lake")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.6)
    # plt.tight_layout()
    plt.show()

    pass


def main():
    compare_gt_vs_rec()
    pass


if __name__ == '__main__':
    main()
    # test1()

    pass
