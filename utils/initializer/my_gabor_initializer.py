import cv2
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


def gabor_filter_opencv(ksize, sigma, theta, lambd, gamma=1.0, psi=0.0, ktype=cv2.CV_32F):
    """
    Generate a 2D Gabor kernel (OpenCV version).

    Parameters:
        sigma σ : Gaussian envelope, set as σ ≈ π/ω
        theta θ : orientation (radians), 8 orientations: θ = (n-1) * π/8, n=1..8
        lambd ω : wavelength (λ = 2π/ω), with ω from 5 frequencies:
                 ω = (π/2) * 2^(-(m-1)/2), m=1..5
        gamma  : spatial aspect ratio, typically 1.0 (isotropic)
        psi   ψ : phase offset, ψ ~ U(0, π)

    Returns:
        2D numpy array of shape ksize×ksize
    """
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)
    return kernel


def get_conv_weight(in_channels, out_channels, kernel_size, device="cpu"):
    """
    Create conv weight tensor initialized with Gabor filters.

    Parameters:
        in_channels  : input channels
        out_channels : output channels
        kernel_size  : kernel size (int)
        device       : torch device

    Returns:
        Tensor of shape [out_channels, in_channels, kH, kW]
    """
    # build gabor library
    gabor_lib = build_gabor_library(kernel_size, multiple=multiple, gamma=gamma)

    # random sample filters for out_channels
    assert len(gabor_lib) >= out_channels, "library too small, increase multiple"
    selected = np.random.choice(len(gabor_lib), size=out_channels, replace=False)

    if isinstance(kernel_size, tuple):
        kH, kW = int(kernel_size[0]), int(kernel_size[1])  # + 解包 tuple
    else:
        kH = kW = int(kernel_size)
    weight = np.zeros((out_channels, in_channels, kH, kW), dtype=np.float32)
    for i, idx in enumerate(selected):
        kernel = gabor_lib[idx]
        for c in range(in_channels):
            weight[i, c] = kernel  # copy kernel to each input channel

    return torch.tensor(weight, dtype=torch.float32, device=device)


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


if __name__ == '__main__':
    main()

    pass
