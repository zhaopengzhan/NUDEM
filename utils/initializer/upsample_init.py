import numpy as np
import torch


def bilinear_kernel(in_channels, out_channels, kernel_size):
    """创建双线性插值的卷积核"""
    if isinstance(kernel_size, tuple):
        kernel_size = int(kernel_size[0])
        
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float32)
    for i in range(in_channels):
        weight[i, i, :, :] = filt
    return torch.from_numpy(weight).permute(1, 0, 2, 3)