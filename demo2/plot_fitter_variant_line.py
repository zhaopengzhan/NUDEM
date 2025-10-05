import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from hydrology.modeling_regressor import AutoRegressor
from hydrology.water_area_curve import HypsometricCurve

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


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

    vmin, vmax = hc_gt.S[10000], hc_gt.S[-10000]
    N_sample = 500
    areas = np.linspace(vmin, vmax, num=N_sample)

    heights_gt = hc_gt.get_height_from_area(areas)

    # TODO: 只改了这个地方
    model = AutoRegressor(areas[::10], heights_gt[::10], 'poly2')
    heights_re = model.get_y_from_x(areas)
    model.show()

    heights_gt = np.diff(heights_gt)
    heights_re = np.diff(heights_re)
    areas = (np.array(areas[:-1]) + np.array(areas[1:])) / 1e6

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
    print(font_size)
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
