import numpy as np
import rasterio
import torch
from matplotlib import pyplot as plt
from numpy.ma import masked_array
from sympy import finite_diff_weights
from torch import nn
from torch.nn import Conv2d,ConvTranspose1d

class FiniteDiffConv(nn.Module):
    def __init__(
            self,
            order: int,
            kernel_size: int,
            type: str,
            stride: int = 1,
            padding: int = 1,
            padding_mode: str = 'reflect',
            h: float = 1.0,  # 栅格间距，非 1 需要按 h^order 缩放
    ):
        super().__init__()
        assert type in {"x", "y", "xy", "lap"}

        self.conv = nn.Conv2d(
            1, 1, kernel_size,
            stride=stride, padding=padding, bias=False, padding_mode=padding_mode
        )

        kernel_weight = self._fd_coeffs_2d(order=order, accuracy=kernel_size - 1, type=type)
        if h != 1.0:
            # 一阶 → /h, 二阶 → /h^2, 混合二阶也 /h^2
            scale = h ** order
            kernel_weight = kernel_weight / scale

        # 将权重拷入 conv，并关闭梯度
        with torch.no_grad():
            self.conv.weight.data.copy_(kernel_weight)
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def _fd_coeffs_1d(self, order: int, accuracy: int) -> torch.Tensor:
        assert accuracy % 2 == 0, "accuracy 必须是偶数，比如 2/4/6 ..."
        m = accuracy // 2
        x_list = list(range(-m, m + 1))  # 对称点
        weights = finite_diff_weights(order, x_list, 0)  # 取该阶数，最后一个子列表
        coeffs = weights[order][-1]
        return torch.tensor(coeffs, dtype=torch.float32)

    def _fd_coeffs_2d(self, order: int, accuracy: int, type: str = "x") -> torch.Tensor:
        coeffs = self._fd_coeffs_1d(order, accuracy)
        k = len(coeffs)

        if type == "x":
            kernel = torch.zeros((k, k), dtype=torch.float32)
            kernel[k // 2, :] = coeffs
        elif type == "y":
            kernel = torch.zeros((k, k), dtype=torch.float32)
            kernel[:, k // 2] = coeffs
        elif type == "xy":
            if order != 2:
                raise ValueError("混合导数仅支持二阶 (order=2)")
            coeffs1 = self._fd_coeffs_1d(1, accuracy)  # 一阶
            kernel = torch.outer(coeffs1, coeffs1)  # 外积
        elif type == "lap":
            if order != 2:
                raise ValueError("Laplacian 仅支持二阶 (order=2)")
            kernel = torch.zeros((k, k), dtype=torch.float32)
            kernel[k // 2, :] = coeffs  # Dxx
            kernel[:, k // 2] += coeffs  # Dyy
        else:
            raise ValueError(f"未知 type={type}")

        return kernel.view(1, 1, k, k)

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

if __name__ == '__main__':
    # main(1, 2)
    type = "y"
    fdconv = FiniteDiffConv(0, 3,padding=1, type=type)
    # path = r'F:\BaiduNetdiskDownload\cdm_contour\AUNDEM.tif'
    path = r'F:\BaiduNetdiskDownload\cdm_contour\小柴旦湖_utm_tmp.tif'

    with rasterio.open(path) as src:
        img = src.read(1)
        nodata = src.nodata

    img_ts = torch.from_numpy(img)
    img_ts = img_ts.unsqueeze(0).unsqueeze(0)
    img_ts = img_ts.to(dtype=torch.float32)
    print(img_ts.shape)

    outputs = fdconv(img_ts)

    print(outputs.shape)
    print(outputs.mean())

    # ---- 掩膜处理 ----
    mask = (img == nodata) if nodata is not None else np.zeros_like(img, dtype=bool)
    img_masked = masked_array(img, mask=mask)
    out_masked = masked_array(outputs, mask=mask)

    # 百分位拉伸
    stretched_img = stretch_percentile(img_masked, p_low=2, p_high=98)
    stretched_out = stretch_percentile(out_masked, p_low=2, p_high=98)

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im0 = axes[0].imshow(stretched_img.squeeze(), cmap='terrain')
    axes[0].set_title("Original DEM (2–98% stretch)")
    axes[0].axis("off")

    im1 = axes[1].imshow(stretched_out.squeeze(), cmap='RdBu')  # 导数结果一般用发散色
    axes[1].set_title("Finite Difference Output (2–98% stretch)")
    axes[1].axis("off")

    plt.suptitle(f"{type} loss {outputs.mean():.3f}", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.show()

    pass
