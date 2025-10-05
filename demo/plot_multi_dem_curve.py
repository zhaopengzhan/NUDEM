import matplotlib.pyplot as plt
import numpy as np
import rasterio

from processor import reproject_raster_by_raster, clip_raster_by_raster, clip_raster_by_ploygon

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def check_dem_alignment(dem_gt_path: str, dem_re_path: str) -> bool:
    ok = True
    with rasterio.open(dem_gt_path) as gt, rasterio.open(dem_re_path) as re:
        # 1) CRS
        if gt.crs == re.crs:
            print("✅ CRS equal:", True)
        else:
            print("❌ CRS equal:", False)
            print("\tGT CRS:", gt.crs)
            print("\tRE CRS:", re.crs)
            ok = False

        # 2) transform
        if gt.transform == re.transform:
            print("✅ Transform equal:", True)
        else:
            print("❌ Transform equal:", False)
            # print("\t GT Transform:", gt.transform)
            # print("\t RE Transform:", re.transform)
            # resolution (always show for reference)
            print("\tResolution GT:", (gt.transform.a, gt.transform.e))
            print("\tResolution RE:", (re.transform.a, re.transform.e))
            ok = False

        # 3) shape & mask
        gt_dem = gt.read(1, masked=True)
        re_dem = re.read(1, masked=True)
        same_shape = gt_dem.shape == re_dem.shape
        same_mask = np.array_equal(gt_dem.mask, re_dem.mask) if same_shape else False

        if same_shape:
            print("✅ Shape equal:", True)
        else:
            print("❌ Shape equal:", False)
            print("\tGT Shape:", gt_dem.shape)
            print("\tRE Shape:", re_dem.shape)
            ok = False

        if same_mask:
            print("✅ Mask equal:", True)
        else:
            print("❌ Mask equal:", False)
            print("\tGT mask valid pixels:", np.count_nonzero(~gt_dem.mask))
            print("\tRE mask valid pixels:", np.count_nonzero(~re_dem.mask))
            ok = False

    return ok


def compare_gt_vs_rec():
    dem_gt_path = r'F:\BaiduNetdiskDownload\cdm_contour\小柴旦湖_utm_tmp.tif'
    dem_re_path = r'F:\BaiduNetdiskDownload\cdm_contour\AUNDEM.tif'

    if check_dem_alignment(dem_gt_path, dem_re_path):
        pass
    else:

        _res1_ds = reproject_raster_by_raster(src=dem_re_path, ref=dem_gt_path, match_resolution=True)

        shp_path = r'F:\BaiduNetdiskDownload\cdm_contour\New Folder\内圈.shp'

        clip_raster_by_ploygon(src_ds=_res1_ds, ref=shp_path, output_path=dem_re_path.replace('.tif', '_align.tif'))

        clip_raster_by_ploygon(src=dem_gt_path, ref=shp_path, output_path=dem_gt_path.replace('.tif', '_align.tif'))

        check_dem_alignment(dem_re_path.replace('.tif', '_align.tif'),
                            dem_gt_path.replace('.tif', '_align.tif'))
        pass

    pass


def main():
    path = r''
    compare_gt_vs_rec()
    pass


def test1():
    dem_gt_path = r'F:\BaiduNetdiskDownload\cdm_contour\小柴旦湖_utm_tmp.tif'
    dem_re_path = r'F:\BaiduNetdiskDownload\cdm_contour\AUNDEM.tif'

    check_dem_alignment(dem_gt_path, dem_re_path)


if __name__ == '__main__':
    main()
    # test1()

    pass
