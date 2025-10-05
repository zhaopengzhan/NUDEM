import json
import warnings

import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
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


def expand_bounds(ref_bounds, scale=0.05):
    # 计算宽度和高度
    width = ref_bounds[2] - ref_bounds[0]
    height = ref_bounds[3] - ref_bounds[1]

    # 计算扩增量 (5%)
    expand_width = width * scale
    expand_height = height * scale

    # 调整边界
    new_xmin = ref_bounds[0] - expand_width / 2
    new_xmax = ref_bounds[2] + expand_width / 2
    new_ymin = ref_bounds[1] - expand_height / 2
    new_ymax = ref_bounds[3] + expand_height / 2

    # 新的边界
    new_bounds = (new_xmin, new_ymin, new_xmax, new_ymax)

    # 创建新的矩形几何对象
    bbox_geom = [box(*new_bounds)]

    return bbox_geom


def clip_raster_by_raster(src=None, ref=None, src_ds=None, ref_ds=None, output_path=None, scale=None,
                          force_match=False):
    """
    Clip a source raster using the bounds of a reference raster.

    Parameters:
    -----------
    src : str, optional
        Path to the source raster file.
    ref : str, optional
        Path to the reference raster file.
    src_ds : rasterio.Dataset, optional
        Opened source raster dataset.
    ref_ds : rasterio.Dataset, optional
        Opened reference raster dataset.
    output_path : str, optional
        Path to save the clipped raster. If None, the result is returned as a MemoryFile dataset.

    Returns:
    --------
    If output_path is provided:
        None (saves the clipped raster to the specified path).
    If output_path is None:
        A rasterio.Dataset object from a MemoryFile.
    """
    # Validate inputs
    if (src is None and src_ds is None) or (ref is None and ref_ds is None):
        raise ValueError("Either src/ref paths or src_ds/ref_ds datasets must be provided.")

    # Open datasets if paths are provided
    if src_ds is None:
        src_ds = rasterio.open(src)
    if ref_ds is None:
        ref_ds = rasterio.open(ref)

    # Get the bounds of the reference raster
    ref_bounds = ref_ds.bounds
    src_bounds = src_ds.bounds

    # 如果坐标系不一致
    if src_ds.crs != ref_ds.crs:
        # print(f"转换坐标系: {ref_ds.crs} → {src_ds.crs}")
        ref_bounds = transform_bounds(ref_ds.crs, src_ds.crs, *ref_bounds)
        bbox_geom = [box(*ref_bounds)]
    else:
        bbox_geom = [box(*ref_bounds)]

    # 如果扩展边界裁剪
    if scale is not None:
        bbox_geom = expand_bounds(ref_bounds, scale)

    # 检查范围是否有交集
    if (
            ref_bounds[2] < src_bounds[0] or  # 参考影像 maxx < 目标影像 minx
            ref_bounds[0] > src_bounds[2] or  # 参考影像 minx > 目标影像 maxx
            ref_bounds[3] < src_bounds[1] or  # 参考影像 maxy < 目标影像 miny
            ref_bounds[1] > src_bounds[3]  # 参考影像 miny > 目标影像 maxy
    ):
        raise ValueError("裁剪区域超出影像范围，无法裁剪。请检查影像坐标系或边界。")

    # Clip the source raster using the reference bounds
    clipped_image, clipped_transform = mask(src_ds, bbox_geom, crop=True)

    # 如果强制匹配，且clip结果长宽不完全相等，那么强制对齐一下
    if force_match and clipped_image.shape[-2:] != ref_ds.shape[-2:]:  # 忽略波段维度
        target_height, target_width = ref_ds.shape[-2:]
        clipped_image = clipped_image[..., :target_height, :target_width]
        clipped_transform = ref_ds.transform

    # Prepare metadata for the clipped raster
    new_meta = src_ds.meta.copy()
    new_meta.update({
        "height": clipped_image.shape[1],
        "width": clipped_image.shape[2],
        "transform": clipped_transform
    })

    # Save to output path or return as MemoryFile dataset
    if output_path:
        with rasterio.open(output_path, "w", **new_meta) as dst:
            dst.write(clipped_image)
        return output_path
    else:
        # Save to MemoryFile
        memfile = MemoryFile()
        with memfile.open(**new_meta) as dst:
            dst.write(clipped_image)
        return memfile.open()


def clip_raster_by_rectangle(src=None, ref=None, src_ds=None, ref_ds=None, output_path=None, scale=None,
                       force_match=False):
    """
    Clip a source raster using the bounds of a reference raster.

    Parameters:
    -----------
    src : str, optional
        Path to the source raster file.
    ref : str, optional
        Path to the reference raster file.
    src_ds : rasterio.Dataset, optional
        Opened source raster dataset.
    ref_ds : rasterio.Dataset, optional
        Opened reference raster dataset.
    output_path : str, optional
        Path to save the clipped raster. If None, the result is returned as a MemoryFile dataset.

    Returns:
    --------
    If output_path is provided:
        None (saves the clipped raster to the specified path).
    If output_path is None:
        A rasterio.Dataset object from a MemoryFile.
    """

    # Validate inputs
    if (src is None and src_ds is None) or (ref is None and ref_ds is None):
        raise ValueError("Either src/ref paths or src_ds/ref_ds datasets must be provided.")

    # Open datasets if paths are provided
    if src_ds is None:
        src_ds = rasterio.open(src)
    if ref_ds is None:
        ref_ds = gpd.read_file(ref)

    # Get the bounds of the reference raster
    src_bounds = src_ds.bounds
    ref_bounds = ref_ds.total_bounds  # shp

    # 如果坐标系不一致
    if src_ds.crs != ref_ds.crs:
        # print(f"转换坐标系: {ref_ds.crs} → {src_ds.crs}")
        ref_bounds = transform_bounds(ref_ds.crs, src_ds.crs, *ref_bounds)
        bbox_geom = [box(*ref_bounds)]
    else:
        bbox_geom = [box(*ref_bounds)]

    # 如果扩展边界裁剪
    if scale is not None:
        bbox_geom = expand_bounds(ref_bounds, scale)

    # 检查范围是否有交集
    if (
            ref_bounds[2] < src_bounds[0] or  # 参考影像 maxx < 目标影像 minx
            ref_bounds[0] > src_bounds[2] or  # 参考影像 minx > 目标影像 maxx
            ref_bounds[3] < src_bounds[1] or  # 参考影像 maxy < 目标影像 miny
            ref_bounds[1] > src_bounds[3]  # 参考影像 miny > 目标影像 maxy
    ):
        raise ValueError("裁剪区域超出影像范围，无法裁剪。请检查影像坐标系或边界。")

    # Clip the source raster using the reference bounds
    clipped_image, clipped_transform = mask(src_ds, bbox_geom, crop=True)

    # 如果强制匹配，且clip结果长宽不完全相等，那么强制对齐一下
    if force_match and clipped_image.shape[-2:] != ref_ds.shape[-2:]:  # 忽略波段维度
        target_height, target_width = ref_ds.shape[-2:]
        clipped_image = clipped_image[..., :target_height, :target_width]
        clipped_transform = ref_ds.transform

    # Prepare metadata for the clipped raster
    new_meta = src_ds.meta.copy()
    new_meta.update({
        "height": clipped_image.shape[1],
        "width": clipped_image.shape[2],
        "transform": clipped_transform
    })

    # Save to output path or return as MemoryFile dataset
    if output_path:
        with rasterio.open(output_path, "w", **new_meta) as dst:
            dst.write(clipped_image)
        return output_path
    else:
        # Save to MemoryFile
        memfile = MemoryFile()
        with memfile.open(**new_meta) as dst:
            dst.write(clipped_image)
        return memfile.open()


def clip_raster_by_ploygon(src=None, ref=None, src_ds=None, ref_ds=None, output_path=None,
                           force_match=False, nodata=0, dtype="float32"):
    """
    This function improves upon the previous `clip_raster_by_shp` implementation.
    The earlier version clipped rasters using only the rectangular bounding box of a shapefile,

    In contrast, this function uses vector geometries
        Supported geometry types:
        - Polygon (currently implemented)

    Parameters:
    -----------
    src : str, optional
        Path to the source raster file.
    ref : str, optional
        Path to the reference raster file.
    src_ds : rasterio.Dataset, optional
        Opened source raster dataset.
    ref_ds : rasterio.Dataset, optional
        Opened reference raster dataset.
    output_path : str, optional
        Path to save the clipped raster. If None, the result is returned as a MemoryFile dataset.

    Returns:
    --------
    If output_path is provided:
        None (saves the clipped raster to the specified path).
    If output_path is None:
        A rasterio.Dataset object from a MemoryFile.
    """

    # Validate inputs
    if (src is None and src_ds is None) or (ref is None and ref_ds is None):
        raise ValueError("Either src/ref paths or src_ds/ref_ds datasets must be provided.")

    # Open datasets if paths are provided
    if src_ds is None:
        src_ds = rasterio.open(src)
    if ref_ds is None:
        ref_ds = gpd.read_file(ref)

    # 如果坐标系不一致
    if src_ds.crs != ref_ds.crs:
        # print(f"转换坐标系: {ref_ds.crs} → {src_ds.crs}")
        ref_ds = ref_ds.to_crs(src_ds.crs)
        polygon_geoms = [json.loads(ref_ds.to_json())["features"][0]["geometry"]]
    else:
        polygon_geoms = [json.loads(ref_ds.to_json())["features"][0]["geometry"]]

    # Clip the source raster using the reference bounds
    clipped_image, clipped_transform = mask(src_ds, polygon_geoms, crop=True)

    # 如果强制匹配，且clip结果长宽不完全相等，那么强制对齐一下
    if force_match and clipped_image.shape[-2:] != ref_ds.shape[-2:]:  # 忽略波段维度
        target_height, target_width = ref_ds.shape[-2:]
        clipped_image = clipped_image[..., :target_height, :target_width]
        clipped_transform = ref_ds.transform

    # Prepare metadata for the clipped raster
    new_meta = src_ds.meta.copy()
    new_meta.update({
        "height": clipped_image.shape[1],
        "width": clipped_image.shape[2],
        "transform": clipped_transform,
        "nodata": nodata,
        "dtype": dtype,
    })

    # Save to output path or return as MemoryFile dataset
    if output_path:
        with rasterio.open(output_path, "w", **new_meta) as dst:
            dst.write(clipped_image)
        return output_path
    else:
        # Save to MemoryFile
        memfile = MemoryFile()
        with memfile.open(**new_meta) as dst:
            dst.write(clipped_image)
        return memfile.open()

