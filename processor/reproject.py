import warnings

import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.warp import transform_bounds
from shapely.geometry import box

warnings.filterwarnings("ignore")
import logging

logging.basicConfig(
    level=logging.INFO,  # 控制最低输出等级
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]  # 输出到控制台
)
print = logging.info



def reproject_raster_by_raster(src=None, ref=None, src_ds=None, ref_ds=None, output_path=None, match_resolution=False):
    # Validate inputs
    if (src is None and src_ds is None) or (ref is None and ref_ds is None):
        raise ValueError("Either src/ref paths or src_ds/ref_ds datasets must be provided.")

    # Open datasets if paths are provided
    if src_ds is None:
        src_ds = rasterio.open(src)
    if ref_ds is None:
        ref_ds = rasterio.open(ref)

    if match_resolution:
        # TODO: key 计算仿射变换矩阵
        transform, width, height = calculate_default_transform(
            src_ds.crs, ref_ds.crs, src_ds.width, src_ds.height, *src_ds.bounds, resolution=ref_ds.res
        )
    else:
        transform, width, height = calculate_default_transform(
            src_ds.crs, ref_ds.crs, src_ds.width, src_ds.height, *src_ds.bounds
        )

    # 更新元数据
    new_meta = src_ds.meta.copy()
    new_meta.update({
        "crs": ref_ds.crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    # Save to output path or return as MemoryFile dataset
    if output_path:
        with rasterio.open(output_path, "w+", **new_meta) as dst:
            for i in range(1, src_ds.count + 1):  # 逐波段重投影
                reproject(
                    source=rasterio.band(src_ds, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src_ds.transform,
                    src_crs=src_ds.crs,
                    dst_transform=transform,
                    dst_crs=ref_ds.crs,
                    resampling=Resampling.nearest  # 可选: bilinear, cubic等
                )
        return output_path
    else:
        # Save to MemoryFile
        memfile = MemoryFile()
        with memfile.open(**new_meta) as dst:
            for i in range(1, src_ds.count + 1):  # 逐波段重投影
                reproject(
                    source=rasterio.band(src_ds, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src_ds.transform,
                    src_crs=src_ds.crs,
                    dst_transform=transform,
                    dst_crs=ref_ds.crs,
                    resampling=Resampling.nearest  # 可选: bilinear, cubic等
                )
        return memfile.open()




def reproject_raster_by_shp(src=None, ref=None, src_ds=None, ref_ds=None, output_path=None):
    # Validate inputs
    if (src is None and src_ds is None) or (ref is None and ref_ds is None):
        raise ValueError("Either src/ref paths or src_ds/ref_ds datasets must be provided.")

    # Open datasets if paths are provided
    if src_ds is None:
        src_ds = rasterio.open(src)
    if ref_ds is None:
        ref_ds = gpd.read_file(ref)  # shp

    # TODO: shp分辨分辨率，去掉了匹配分辨率环节
    transform, width, height = calculate_default_transform(
        src_ds.crs, ref_ds.crs, src_ds.width, src_ds.height, *src_ds.bounds
    )

    # 更新元数据
    new_meta = src_ds.meta.copy()
    new_meta.update({
        "crs": ref_ds.crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    # Save to output path or return as MemoryFile dataset
    if output_path:
        with rasterio.open(output_path, "w+", **new_meta) as dst:
            for i in range(1, src_ds.count + 1):  # 逐波段重投影
                reproject(
                    source=rasterio.band(src_ds, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src_ds.transform,
                    src_crs=src_ds.crs,
                    dst_transform=transform,
                    dst_crs=ref_ds.crs,
                    resampling=Resampling.nearest  # 可选: bilinear, cubic等
                )
        return output_path
    else:
        # Save to MemoryFile
        memfile = MemoryFile()
        with memfile.open(**new_meta) as dst:
            for i in range(1, src_ds.count + 1):  # 逐波段重投影
                reproject(
                    source=rasterio.band(src_ds, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src_ds.transform,
                    src_crs=src_ds.crs,
                    dst_transform=transform,
                    dst_crs=ref_ds.crs,
                    resampling=Resampling.nearest  # 可选: bilinear, cubic等
                )
        return memfile.open()
