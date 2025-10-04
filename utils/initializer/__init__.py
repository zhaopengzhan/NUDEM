import math

import torch
import torch.nn as nn
import torch.nn.init as init

from .gabor_initializer import get_conv_weight
from .landsat_initializer import get_naip_conv_weight, get_landsat_conv_weight
from .upsample_init import bilinear_kernel


def init_conv(m: nn.Module,
              kind: str = "kaiming_uniform",
              # "kaiming_uniform" | "kaiming_normal" | "xavier_uniform" | "xavier_normal" | "orthogonal" | "dirac" | "trunc_normal"
              nonlinearity: str = "relu",  # 影响 gain（如 "relu", "leaky_relu", "tanh", "sigmoid"...）
              fan_mode: str = "fan_in",  # 仅 Kaiming: "fan_in" 或 "fan_out"
              leaky_neg_slope: float = 0.0,  # 仅 LeakyReLU: a
              bias_uniform: bool = True):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        w = m.weight
        if kind == "kaiming_uniform":
            init.kaiming_uniform_(w, a=leaky_neg_slope, mode=fan_mode,
                                  nonlinearity="leaky_relu" if leaky_neg_slope > 0 else nonlinearity)
        elif kind == "kaiming_normal":
            init.kaiming_normal_(w, a=leaky_neg_slope, mode=fan_mode,
                                 nonlinearity="leaky_relu" if leaky_neg_slope > 0 else nonlinearity)
        elif kind == "xavier_uniform":
            init.xavier_uniform_(w, gain=init.calculate_gain(nonlinearity))
        elif kind == "xavier_normal":
            init.xavier_normal_(w, gain=init.calculate_gain(nonlinearity))
        elif kind == "orthogonal":
            init.orthogonal_(w, gain=init.calculate_gain(nonlinearity))
        elif kind == "dirac":  # 适合 stride=1，groups 可用作深度可分离/保通道恒等
            init.dirac_(w)
        elif kind == "trunc_normal":
            init.trunc_normal_(w, mean=0.0, std=0.02)  # ViT 常用 0.02
        elif kind == "gabor_init":
            with torch.no_grad():
                w1 = get_conv_weight(m.in_channels, m.out_channels, kernel_size=m.kernel_size, multiple=20,
                                     device=w.device)
                m.weight.data.copy_(w1)
        elif kind == "bilinear_init":
            with torch.no_grad():
                w1 = bilinear_kernel(m.in_channels, m.out_channels, kernel_size=m.kernel_size)
                m.weight.data.copy_(w1)
        elif kind == "naip_init":
            with torch.no_grad():
                w1 = get_naip_conv_weight(m.out_channels, kernel_size=m.kernel_size)[:, [2, 1, 0, 3], ...]
                # w1 = get_naip_conv_weight(m.out_channels, kernel_size=m.kernel_size)

                m.weight.data.copy_(w1)
        elif kind == "landsat_init":
            with torch.no_grad():
                w1 = get_landsat_conv_weight(m.out_channels, kernel_size=m.kernel_size)
                if m.in_channels == 4:
                    w1 = get_landsat_conv_weight(m.out_channels, kernel_size=m.kernel_size)[:, [1, 2, 3, 4], ...]
                m.weight.data.copy_(w1)
        else:
            raise ValueError(f"Unknown init kind: {kind}")

        if m.bias is not None:
            if bias_uniform:
                fan_in = w.shape[1] * w.shape[2] * w.shape[3] if w.dim() == 4 else w.numel()
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
            else:
                init.zeros_(m.bias)

# 用法：对整网递归应用
# model.apply(lambda m: init_conv(m, kind="kaiming_uniform", nonlinearity="relu", fan_mode="fan_in"))
