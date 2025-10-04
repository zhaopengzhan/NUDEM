import os
import warnings
import logging

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
print = logging.info


def mosaic_tiles(tile_dir, out_path, method="average"):
    """
    将小块影像拼接为一个大图，自动根据地理坐标对齐，解决重叠问题。

    Parameters
    ----------
    tile_dir : str
        小块文件夹路径
    out_path : str
        输出文件路径
    method : str
        重叠区域的处理方式：
        - "last"    : 最后一个覆盖前一个 (默认 rasterio 行为)
        - "min"     : 取最小值
        - "max"     : 取最大值
        - "average" : 平均值
    """
    tile_paths = glob.glob(os.path.join(tile_dir, "*.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"未找到影像小块: {tile_dir}")

    print(f"找到 {len(tile_paths)} 个小块，开始拼接...")

    src_files_to_mosaic = []
    for p in tile_paths:
        src = rasterio.open(p)
        src_files_to_mosaic.append(src)

    def average(values, *args, **kwargs):
        print(args, kwargs)
        valid = values[values != 0]  # 忽略0当作nodata
        if len(valid) == 0:
            return 0
        return np.mean(valid)

    # merge 会自动根据坐标拼接
    # mosaic, out_trans = merge(src_files_to_mosaic, method=method)
    mosaic, out_trans = merge(src_files_to_mosaic, method=average)

    # 更新元数据
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"✅ 拼接完成，输出: {out_path} (大小: {mosaic.shape[2]}×{mosaic.shape[1]})")

    # 关闭 dataset
    for src in src_files_to_mosaic:
        src.close()


if __name__ == "__main__":
    tile_dir = r"东城区_小块"
    out_path = r"东城区_mosaic.tif"




    # 方法可选： "last" | "min" | "max" | "average"
    mosaic_tiles(tile_dir, out_path, method="first")
