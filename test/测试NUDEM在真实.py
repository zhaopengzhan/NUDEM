import glob

import numpy as np
import rasterio
import torch

from models.nudem.modeling_nudem import NUDEM


def main():
    init_path = r"/data/lyf/DeepZLab/projects/NUDEM/data2/ref/xiaochaidan_mask_30m.tif"
    contour_path = r'/data/lyf/DeepZLab/projects/NUDEM/data2/ref/kept_lnds.tif'
    rel_contour_path = r'/data/lyf/DeepZLab/projects/NUDEM/data2/run6_samegrid/*.tif'

    vmin, vmax = 3173.0, 3179.5
    scale = vmax - vmin

    # 1. dem
    with rasterio.open(init_path) as src:
        img = src.read(1)

        valid_mask = np.where(img != src.nodata, 1, 0).astype(np.bool)
        img_norm = np.full_like(img, src.nodata, dtype=np.float32)
        img_norm[valid_mask] = (img[valid_mask] - vmin) / scale

    init_tensor = torch.as_tensor(img_norm, dtype=torch.float32)
    init_tensor = torch.randn_like(init_tensor, dtype=torch.float32)

    # 2. 有值等高线
    with rasterio.open(contour_path) as src:
        contour_gt = src.read(1)
        # contour_gt = (contour_gt - vmin) / scale
        contour_gt_nodata = src.nodata

    cons_mask_np = (contour_gt != contour_gt_nodata)

    # + ===== 按固定区间 [3174, 3179] 归一化 =====

    contour_gt_norm = np.full_like(contour_gt, contour_gt_nodata, dtype=np.float32)
    contour_gt_norm[cons_mask_np] = (contour_gt[cons_mask_np] - vmin) / scale

    cons_vals_np = np.where(cons_mask_np, contour_gt_norm, contour_gt_nodata).astype("float32")

    # 3. 无值等高线
    not_picked = glob.glob(rel_contour_path)
    unknown_masks = []
    for path in not_picked:
        with rasterio.open(path) as src:
            _unknown_masks = src.read(1)
            unknown_masks.append(_unknown_masks)
    masks_stacked = np.stack(unknown_masks, 0)
    masks_stacked = torch.as_tensor(masks_stacked, dtype=torch.bool)

    model = NUDEM(
        init_tensor,
        torch.from_numpy(valid_mask),
        torch.from_numpy(cons_vals_np),
        torch.from_numpy(cons_mask_np),
        masks_stacked,
        kernel_size=3,
    ).cuda(1)
    model.fit(max_epoch=8000, epoch_print_result=500)
    pass


if __name__ == "__main__":
    main()
    pass
