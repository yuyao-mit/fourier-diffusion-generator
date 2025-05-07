# ssim.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

@torch.no_grad()
def gaussian(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size).float()
    coords -= window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

@torch.no_grad()
def create_window(window_size: int, channel: int, device=None, dtype=None) -> torch.Tensor:
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window.to(device=device, dtype=dtype)

def _ssim(img1, img2, window, window_size, channel, reduction='mean'):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'none':
        return ssim_map
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")

class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, reduction: str = 'mean'):
        """
        Structural Similarity Index Measure (SSIM) loss module.

        Args:
            window_size (int): Size of the sliding window.
            reduction (str): 'mean' or 'none'.
        """
        super().__init__()
        self.window_size = window_size
        self.reduction = reduction

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same dimensions")

        _, channel, _, _ = img1.shape
        window = create_window(
            window_size=self.window_size,
            channel=channel,
            device=img1.device,
            dtype=img1.dtype
        )
        return _ssim(img1, img2, window, self.window_size, channel, self.reduction)

def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    reduction: str = 'mean'
) -> torch.Tensor:
    _, channel, _, _ = img1.shape
    window = create_window(window_size, channel, device=img1.device, dtype=img1.dtype)
    return _ssim(img1, img2, window, window_size, channel, reduction)
