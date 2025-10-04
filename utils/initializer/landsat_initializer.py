import cv2
import numpy as np
import torch.nn as nn

Landsat8_OLI = {
    "B1": {"lambda_nm": 443, "bandwidth_nm": 16},  # Coastal/Aerosol
    "B2": {"lambda_nm": 482, "bandwidth_nm": 60},  # Blue
    "B3": {"lambda_nm": 561, "bandwidth_nm": 57},  # Green
    "B4": {"lambda_nm": 655, "bandwidth_nm": 37},  # Red
    "B5": {"lambda_nm": 865, "bandwidth_nm": 28},  # NIR
    "B6": {"lambda_nm": 1609, "bandwidth_nm": 85},  # SWIR1
    "B7": {"lambda_nm": 2201, "bandwidth_nm": 187},  # SWIR2
}


def gabor_filter_opencv(ksize, sigma, theta, lambd, gamma=1.0, psi=0.0, ktype=cv2.CV_32F):
    """
    Generate a 2D Gabor kernel (OpenCV version).

    Parameters:

        theta θ : orientation (radians), 8 orientations: θ = (n-1) * π/8, n=1..8
        lambd ω : wavelength (λ = 2π/ω), with ω from 5 frequencies:
            ω = (π/2) * 2^(-(m-1)/2), m=1..5
        psi   ψ : phase offset, ψ ~ U(0, π)
        sigma σ : Gaussian envelope, set as σ ≈ π/ω
        gamma  : spatial aspect ratio, typically 1.0 (isotropic)
    Returns:
        2D numpy array of shape ksize×ksize
    """
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)
    return kernel


def rho_from_r_linear(r, r0=0.05, r1=0.25, rho_min=0.45, rho_max=0.85, do_clip=True):
    """公式(1) 实现：r -> ρ 的线性映射（可选硬夹紧）"""
    t = (r - r0) / (r1 - r0)
    rho = rho_min + (rho_max - rho_min) * t
    if do_clip:
        rho = max(rho_min, min(rho, rho_max))
    return rho


def sigma_from_rho_omega(rho, omega, ksize, ):
    """用 σ = (2π·ρ)/ω，把 ρ 和 ω 化成 σ；可对 3×3 做数值安全裁剪"""
    sigma = (2.0 * math.pi * rho) / max(1e-8, omega)
    if ksize <= 3:
        sigma = max(0.55, min(sigma, 1.25))
    elif ksize <= 7:
        sigma = max(1.0, min(sigma, 3.0))
    elif ksize <= 11:
        sigma = max(2.0, min(sigma, 5.0))
    return sigma


def get_sigma_from_bandwith(ksize, lambda_c, bw, omega,
                            r0=0.05, r1=0.25,
                            rho_min=0.45, rho_max=0.85,
                            do_clip=True):
    """
    主调函数：输入 r 和 ω，返回稳定的 σ
    1) r -> ρ (带宽比)
    2) ρ + ω -> σ
    """
    r = bw / lambda_c
    rho = rho_from_r_linear(r, r0, r1, rho_min, rho_max, do_clip)
    sigma = sigma_from_rho_omega(rho, omega, ksize)
    return sigma


def vdc_base2(n: int) -> np.ndarray:
    """
    生成前 n 个 Van der Corput 序列 (base=2)，范围 [0,1)。
    """
    seq = np.zeros(n)
    for i in range(n):
        x, f, k = 0.0, 0.5, i
        while k > 0:
            x += f * (k & 1)
            k >>= 1
            f *= 0.5
        seq[i] = x
    return seq


def idx_to_angles(n: int) -> np.ndarray:
    """
    输入 n，返回前 n 个角度序列 (弧度，范围 [0, π) )
    """
    seq = vdc_base2(n)
    return seq * math.pi


def get_conv_weight(sensors_dict, out_channels, kernel_size, gamma=1.0, device="cpu"):
    if isinstance(kernel_size, tuple):
        kH, kW = int(kernel_size[0]), int(kernel_size[1])
    else:
        kH = kW = int(kernel_size)

    W_list = []
    for idx, cfg in enumerate(sensors_dict.values()):
        lambda_nm = cfg['lambda_nm']
        bandwidth_nm = cfg['bandwidth_nm']

        freq = 1.0 / (0.005 * lambda_nm)  # 条纹频率 ~ 1/λ
        sigma = get_sigma_from_bandwith(ksize=kW, lambda_c=lambda_nm, bw=bandwidth_nm, omega=freq)
        # sigma = 1.0
        theta_list = idx_to_angles(out_channels)

        one_channel_weight = []
        for theta in theta_list:
            gabor_weight = gabor_filter_opencv(ksize=kW, sigma=sigma, theta=theta, lambd=freq, gamma=gamma)
            one_channel_weight.append(gabor_weight)
        W = torch.tensor(np.stack(one_channel_weight), dtype=torch.float32)
        W_list.append(W)
    final_weight = torch.tensor(np.stack(W_list), dtype=torch.float32)

    return torch.tensor(final_weight, dtype=torch.float32, device=device)


def main():
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, bias=False)

    # 1. Initialize Conv2d
    w = get_conv_weight(3, 64, kernel_size=11, multiple=2, device="cpu")
    conv.weight.data.copy_(w)

    print(conv.weight.shape)  # [64, 3, 11, 11]

    # 2. Visualize 25 randomly selected kernels (first input channel only)
    num_show = 36
    idx = np.random.choice(conv.weight.shape[0], num_show, replace=False)

    fig, axes = plt.subplots(6, 6, figsize=(10, 10))
    for ax, i in zip(axes.flat, idx):
        kernel = conv.weight[i, 0].detach().cpu().numpy()
        ax.imshow(kernel, cmap="gray")
        ax.axis("off")
        ax.set_title(f"#{i}", fontsize=8)

    plt.tight_layout()
    plt.show()
    pass


def main1():
    w = get_conv_weight(sensors_dict=Landsat8_OLI, out_channels=64, kernel_size=31)
    print(w.shape)
    w = w.permute(1, 0, 2, 3)

    # 2. Visualize 25 randomly selected kernels (first input channel only)
    num_show = 36
    idx = np.random.choice(w.shape[0], num_show, replace=False)

    fig, axes = plt.subplots(6, 6, figsize=(10, 10))
    for ax, i in zip(axes.flat, idx):
        kernel = w[i, 0].detach().cpu().numpy()
        ax.imshow(kernel, cmap="gray")
        ax.axis("off")
        ax.set_title(f"#{i}", fontsize=8)

    plt.tight_layout()
    plt.show()


NAIP = {
    "B1": {"lambda_nm": 480, "bandwidth_nm": 60},  # Blue
    "B2": {"lambda_nm": 550, "bandwidth_nm": 80},  # Green
    "B3": {"lambda_nm": 660, "bandwidth_nm": 60},  # Red
    "B4": {"lambda_nm": 830, "bandwidth_nm": 120},  # NIR
}


def get_landsat_conv_weight(out_channels, kernel_size):
    return get_conv_weight(sensors_dict=Landsat8_OLI, out_channels=out_channels, kernel_size=kernel_size).permute(1, 0, 2, 3)


def get_naip_conv_weight(out_channels, kernel_size):
    return get_conv_weight(sensors_dict=NAIP, out_channels=out_channels, kernel_size=kernel_size).permute(1, 0, 2, 3)


def main2():
    w1 = get_conv_weight(sensors_dict=Landsat8_OLI, out_channels=64, kernel_size=31)
    print(w1.shape)
    w1 = w1.permute(1, 0, 2, 3)

    w2 = get_conv_weight(sensors_dict=NAIP, out_channels=64, kernel_size=31)
    print(w2.shape)
    w2 = w2.permute(1, 0, 2, 3)

    # 2. Visualize 25 randomly selected kernels (first input channel only)
    num_show = 36
    idx = np.random.choice(w.shape[0], num_show, replace=False)

    fig, axes = plt.subplots(6, 6, figsize=(10, 10))
    for ax, i in zip(axes.flat, idx):
        kernel = w[i, 0].detach().cpu().numpy()
        ax.imshow(kernel, cmap="gray")
        ax.axis("off")
        ax.set_title(f"#{i}", fontsize=8)

    plt.tight_layout()
    plt.show()

    pass


import math, numpy as np, torch, matplotlib.pyplot as plt


# 假定你已有这两个：get_conv_weight(...) 和两个字典 Landsat8_OLI, NAIP
# get_conv_weight 返回形状: [out_channels, in_channels, k, k]

def bands_in_order(dct: dict):
    """保持你在字典里写入的顺序"""
    return list(dct.keys())


def choose_out_indices(out_channels, num_show=8, seed=42):
    rng = np.random.default_rng(seed)
    if num_show >= out_channels:
        return list(range(out_channels))
    return sorted(rng.choice(out_channels, size=num_show, replace=False).tolist())


def plot_compare_kernels(w1, bands1, w2, bands2, common_bands, out_indices, sensor_name_1="Landsat8",
                         sensor_name_2="NAIP"):
    """
    w1, w2: 张量，[in, out, k, k] （注意：我们会把 get_conv_weight 的结果 permute 成这个形状来用）
    bands1, bands2: 各自的 band 名称列表（顺序要和 in 维一致）
    common_bands: 要对齐并展示的 band 名称列表
    out_indices: 要展示的相同 out ch 索引列表
    """
    kH, kW = w1.shape[-2:]
    rows = 2 * len(common_bands)  # 每个 band 两行（w1 在上，w2 在下）
    cols = len(out_indices)

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.9 * rows))
    if rows == 1: axes = np.array([axes])  # 兼容 1 行情况
    if cols == 1: axes = axes[:, None]  # 兼容 1 列情况

    for bi, band in enumerate(common_bands):
        i1 = bands1.index(band)
        i2 = bands2.index(band)
        for cj, oc in enumerate(out_indices):
            # 取核：w[band_idx, out_idx, :, :]
            k1 = w1[i1, oc].detach().cpu().numpy()
            k2 = w2[i2, oc].detach().cpu().numpy()

            ax1 = axes[2 * bi, cj]
            ax2 = axes[2 * bi + 1, cj]

            ax1.imshow(k1, cmap="gray")
            ax1.set_title(f"{sensor_name_1} | {band} | out#{oc}", fontsize=9)
            ax1.axis("off")

            ax2.imshow(k2, cmap="gray")
            ax2.set_title(f"{sensor_name_2} | {band} | out#{oc}", fontsize=9)
            ax2.axis("off")

    plt.tight_layout()
    plt.show()


def main3():
    # 1) 生成两套权重
    w1 = get_conv_weight(sensors_dict=Landsat8_OLI, out_channels=64, kernel_size=31)  # [64, C1, k, k]
    w2 = get_conv_weight(sensors_dict=NAIP, out_channels=64, kernel_size=31)  # [64, C2, k, k]

    # 2) 为了“按 band 再按 out idx”方便，换到 [in, out, k, k]
    # w1 = w1.permute(1, 0, 2, 3).contiguous()
    # w2 = w2.permute(1, 0, 2, 3).contiguous()
    print("w1:", tuple(w1.shape), "w2:", tuple(w2.shape))  # (Cin1, 64, k, k) / (Cin2, 64, k, k)

    bands1 = bands_in_order(Landsat8_OLI)
    bands2 = bands_in_order(NAIP)

    # 3) 尝试用“相同 band 名称”对齐；若无交集，则按索引对齐
    common_bands = [b for b in bands1 if b in bands2]
    if not common_bands:
        min_c = min(len(bands1), len(bands2))
        common_bands = [f"idx{idx}" for idx in range(min_c)]
        # 为索引对齐构造“虚拟名称”，并把 bands1/bands2 都映射成 idx*
        bands1 = [f"idx{idx}" for idx in range(len(bands1))]
        bands2 = [f"idx{idx}" for idx in range(len(bands2))]

    # 4) 选一些 out-channel 索引（相同的 idx 在两边取核）
    out_indices = choose_out_indices(out_channels=w1.shape[1], num_show=8, seed=123)

    # 5) 画图（每个 band 两行：上=w1，下=w2；列=不同 out#）
    plot_compare_kernels(
        w1, bands1, w2, bands2,
        common_bands=common_bands,
        out_indices=out_indices,
        sensor_name_1="Landsat8_OLI",
        sensor_name_2="NAIP"
    )


if __name__ == "__main__":
    main3()

    pass
