import os
import warnings
import logging
import rasterio
from rasterio.windows import Window
from rasterio.io import DatasetReader

warnings.filterwarnings("ignore")

# ========= Logging 设置 =========
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
print = logging.info


def sample_raster_to_tiles(src_path, out_dir, tile_size=1024, overlap=0, prefix="tile"):
    """
    将大图分割成固定大小的小块 (tile_size × tile_size)，支持重叠与边界对齐。

    Parameters
    ----------
    src_path : str
        输入大图路径
    out_dir : str
        小块输出文件夹
    tile_size : int
        每个小块的边长 (像素)
    overlap : int
        重叠大小 (像素)，0 表示无重叠
    prefix : str
        输出文件名前缀
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with rasterio.open(src_path) as src:  # type: DatasetReader
        img_width = src.width
        img_height = src.height
        meta = src.meta.copy()

        print(f"输入影像大小: {img_width} × {img_height}, 波段: {src.count}")

        # 实际步长
        stride = tile_size - overlap
        if stride <= 0:
            raise ValueError("overlap 必须小于 tile_size，否则无法裁剪")

        tile_id = 0
        y = 0
        while y + tile_size <= img_height:
            x = 0
            while x + tile_size <= img_width:
                window = Window(x, y, tile_size, tile_size)
                transform = src.window_transform(window)
                tile = src.read(window=window)

                meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": transform
                })

                out_name = os.path.join(out_dir, f"{prefix}_{tile_id:05d}.tif")
                with rasterio.open(out_name, "w", **meta) as dst:
                    dst.write(tile)

                print(f"保存: {out_name} (位置 x={x}, y={y})")
                tile_id += 1

                x += stride

            # ===== 处理右边界 =====
            if x < img_width:
                x = img_width - tile_size
                if x >= 0:
                    window = Window(x, y, tile_size, tile_size)
                    transform = src.window_transform(window)
                    tile = src.read(window=window)

                    meta.update({
                        "height": tile_size,
                        "width": tile_size,
                        "transform": transform
                    })

                    out_name = os.path.join(out_dir, f"{prefix}_{tile_id:05d}.tif")
                    with rasterio.open(out_name, "w", **meta) as dst:
                        dst.write(tile)
                    print(f"保存: {out_name} (右边界 x={x}, y={y})")
                    tile_id += 1

            y += stride

        # ===== 处理下边界 =====
        if y < img_height:
            y = img_height - tile_size
            if y >= 0:
                x = 0
                while x + tile_size <= img_width:
                    window = Window(x, y, tile_size, tile_size)
                    transform = src.window_transform(window)
                    tile = src.read(window=window)

                    meta.update({
                        "height": tile_size,
                        "width": tile_size,
                        "transform": transform
                    })

                    out_name = os.path.join(out_dir, f"{prefix}_{tile_id:05d}.tif")
                    with rasterio.open(out_name, "w", **meta) as dst:
                        dst.write(tile)
                    print(f"保存: {out_name} (下边界 x={x}, y={y})")
                    tile_id += 1

                    x += stride

                # 右下角补最后一个
                if x < img_width:
                    x = img_width - tile_size
                    window = Window(x, y, tile_size, tile_size)
                    transform = src.window_transform(window)
                    tile = src.read(window=window)

                    meta.update({
                        "height": tile_size,
                        "width": tile_size,
                        "transform": transform
                    })

                    out_name = os.path.join(out_dir, f"{prefix}_{tile_id:05d}.tif")
                    with rasterio.open(out_name, "w", **meta) as dst:
                        dst.write(tile)
                    print(f"保存: {out_name} (右下角 x={x}, y={y});shape=({tile_size},{tile_size})")
                    tile_id += 1

    print(f"✅ 共生成 {tile_id} 个小块")


if __name__ == "__main__":
    # ===== 用户配置 =====
    path = r"F:\水经注下载\东城区\东城区_大图\L18\东城区.tif"
    out_dir = r"东城区_小块"

    sample_raster_to_tiles(path, out_dir, tile_size=1024, overlap=256, prefix="dongcheng")
