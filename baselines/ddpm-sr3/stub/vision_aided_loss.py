"""Constant stand-in for vision_aided_loss.Discriminator; stage 1 builds it but never uses it (lambda_gan = 0)."""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, cv_type='clip', loss_type='multilevel_sigmoid_s', device='cuda'):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.cv_ensemble = nn.Module()

    def forward(self, x, for_G=False, for_real=False):
        return self.dummy.expand(x.shape[0])
