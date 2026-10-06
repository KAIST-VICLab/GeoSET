"""Pure-PyTorch SoftPool2d / soft_pool2d (replaces the SoftPool CUDA extension imported by E3Diff)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_pool2d(x, kernel_size=2, stride=None):
    if stride is None:
        stride = kernel_size
    e = torch.exp(x)
    return F.avg_pool2d(x * e, kernel_size, stride) / F.avg_pool2d(e, kernel_size, stride).clamp_min(1e-12)


class SoftPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return soft_pool2d(x, self.kernel_size, self.stride)
