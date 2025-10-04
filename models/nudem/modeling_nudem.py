import gc
import os
import random

import numpy as np
import rasterio
import torch

from matplotlib import pyplot as plt
from torch.backends import cudnn
from timm.scheduler import CosineLRScheduler
from torch import nn
from tqdm import tqdm

from utils.misc import Wrapper, AverageMeter
from val.val import val_one_epoch
from val.val_H2A_line import val_one_epoch_H2A
from .finite_diff_conv import FiniteDiffConv
from utils.misc import calRunTimer


def stretch_percentile(arr_masked, p_low=2, p_high=98):
    """
    对掩膜数组进行百分位拉伸，统计时自动排除 nodata（被 mask 的像元）
    返回 [0,1] 归一化后的掩膜数组
    """
    # 压缩掉被 mask 的值，仅用有效像元计算分位数
    data = arr_masked.compressed()
    if data.size == 0:
        raise ValueError("有效像元为空，检查数据或 nodata 设置")
    lo, hi = np.percentile(data, [p_low, p_high])
    if hi <= lo:  # 防止极端情况下分母为0
        lo, hi = np.min(data), np.max(data)
        if hi == lo:
            # 全图常数的极端情况
            out = (arr_masked - lo)
            out = out.filled(np.nan)
            return out

    stretched = (arr_masked - lo) / (hi - lo)
    stretched = np.clip(stretched, 0, 1)
    return stretched


def plot(img, nodata):
    mask = (img == nodata) | ~np.isfinite(img)  # + 也顺带屏蔽 nan/inf
    img = np.ma.array(img, mask=mask)

    # 百分位拉伸（排除 nodata）
    stretched = stretch_percentile(img, p_low=2, p_high=98)

    # 可视化：掩膜像元（nodata）自动透明
    plt.figure(figsize=(7, 6))
    im = plt.imshow(stretched, cmap='terrain')  # 不指定颜色，灰度展示
    # plt.colorbar(im, fraction=0.046, pad=0.04, label='Normalized reflectance (2–98% stretch)')
    title = f'Percentile Stretch (2–98%) — nodata transparent'
    if nodata is not None:
        title += f'  | nodata={nodata}'
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()


class NUDEM(nn.Module):

    def __init__(
            self,
            dem: torch.Tensor,
            valid_mask: torch.Tensor,
            abs_val: torch.Tensor,
            abs_mask: torch.Tensor,
            rel_mask: torch.Tensor,
            kernel_size: int = 5
    ):
        super().__init__()
        K, H, W = rel_mask.shape
        self.dem = nn.Parameter(dem)
        self.register_buffer("valid_mask", valid_mask, persistent=False)

        self.register_buffer("abs_val", abs_val, persistent=False)
        self.register_buffer("abs_mask", abs_mask, persistent=False)

        self.rel_h = nn.Parameter(torch.rand(K))
        self.register_buffer("rel_mask", rel_mask, persistent=False)

        self.fd_xx = FiniteDiffConv(order=2, kernel_size=kernel_size, type="x", padding=int(kernel_size // 2))
        self.fd_yy = FiniteDiffConv(order=2, kernel_size=kernel_size, type="y", padding=int(kernel_size // 2))
        self.fd_xy = FiniteDiffConv(order=2, kernel_size=kernel_size, type="xy", padding=int(kernel_size // 2))

        pass

    def thin_plate_loss(self):
        # Thin Plate Spline
        # 计算二阶导
        x = self.dem[None, None, :, :]

        dxx = self.fd_xx(x)
        dyy = self.fd_yy(x)
        dxy = self.fd_xy(x)

        # 薄板能量：∫ (f_xx^2 + 2 f_xy^2 + f_yy^2) ≈ 像素和
        e = dxx.pow(2) + 2.0 * dxy.pow(2) + dyy.pow(2)  # [1,1,H,W]
        e = e[0, 0]

        # 只在有效像元累积
        m = self.valid_mask
        # m.any() 会检查 mask 里是不是有 至少一个 True。如果全是 False（比如整个 DEM 都是 nodata），那 e[m] 就是空的，e[m].mean() 会报错。
        return (e[m].mean() if m.any() else e.mean())

    def abs_contour_mse_loss(self):
        diff = (self.dem[self.abs_mask] - self.abs_val[self.abs_mask])
        return (diff.pow(2).mean())

    def rel_contour_mse_loss(self, return_rel_contour=False):
        rel_val = torch.einsum("kij,k->kij", self.rel_mask, self.rel_h)  # K H W

        # TODO: 这块可能也得修一下，未来等高线重叠可能出问题，重叠应该mean不应该amax，nodata也不一定总是0，会导致mask错误
        rel_val = rel_val.amax(0)  # 1 H W
        _valid_mask = rel_val != 0

        diff = (self.dem[_valid_mask] - rel_val[_valid_mask])
        if return_rel_contour:
            return (diff.pow(2).mean()), rel_val
        return (diff.pow(2).mean())

    def forward(self):
        print("请使用 model.fit()")
        pass

    def fit(
            self,
            max_epoch=2000,
            random_state=42,
            lr=5e-3,
            epoch_print_result = 100

    ):
        gc.collect()
        torch.cuda.empty_cache()
        cudnn.benchmark = True
        cudnn.deterministic = False
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)

        optimizer = torch.optim.Adam([
            {"params": [self.dem], "weight_decay": 0.0},
            {"params": [self.rel_h], "weight_decay": 1e-4}, ],
            lr=lr,
            # weight_decay=1e-8,
            betas=(0.9, 0.999),
            amsgrad=False,
        )

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=max_epoch,
            lr_min=lr * 1e-1,
        )

        loss_record = Wrapper()
        loss_record.register('loss_tps', AverageMeter())
        loss_record.register('loss_abs', AverageMeter())
        loss_record.register('loss_rel', AverageMeter())

        pbar = tqdm(range(max_epoch))
        for epoch in pbar:
            pbar.set_description(f"Epoch:{epoch} ")
            scheduler.step(epoch)

            # 1.有值等高线拉进一下，然后算一下平滑
            optimizer.zero_grad(set_to_none=True)
            loss_abs = self.abs_contour_mse_loss()
            loss_tps = self.thin_plate_loss()
            (loss_abs + loss_tps).backward()
            optimizer.step()

            # 2.无值等高线拉近一下，然后算一下平滑
            optimizer.zero_grad(set_to_none=True)
            loss_rel, rel_contour = self.rel_contour_mse_loss(return_rel_contour=True)
            loss_tps = self.thin_plate_loss()
            (loss_rel + loss_tps).backward()
            optimizer.step()

            loss_record['loss_tps'].update(loss_tps.item(), 1)
            loss_record['loss_abs'].update(loss_abs.item(), 1)
            loss_record['loss_rel'].update(loss_rel.item(), 1)

            loss_dict = {key: f'{loss_record[key].val:.3f}({loss_record[key].avg:.3f})' for key in loss_record.keys()}

            pbar.set_postfix({**loss_dict})

            if epoch % epoch_print_result == 0:
                rel_contour = torch.where(self.valid_mask, rel_contour, -1)
                plot(rel_contour.detach().cpu().numpy(), -1)
                dem = torch.where(self.valid_mask, self.dem, 0.0)
                plot(dem.detach().cpu().numpy(), 0.0)

                path = r"/data/lyf/DeepZLab/projects/NUDEM/data1/小柴旦湖_utm_tmp.tif"
                with rasterio.open(path) as src:
                    profile = src.profile

                profile['count'] = 1
                profile['nodata'] = 0
                out_path = os.path.join('/data/lyf/DeepZLab/projects/NUDEM/run', f'epoch_{epoch}.tif')
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(dem.detach().cpu().numpy(), 1)

                val_one_epoch(pred_dem_path = out_path)
                val_one_epoch_H2A(dem_pred_path = out_path)
        pass
