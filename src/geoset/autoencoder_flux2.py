# Copyright (c) Black Forest Labs.
# Copyright (c) 2026 The GeoSET Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from black-forest-labs/flux2 (Apache-2.0). Modified by the GeoSET authors.
"""Frozen FLUX.2 autoencoder with the packed (2x2 space-to-depth), BatchNorm-normalised 128-channel latent."""

from __future__ import annotations

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch import Tensor, nn


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q, k, v = (rearrange(t, "b c h w -> b 1 (h w) c").contiguous() for t in (q, k, v))
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        return x + self.proj_out(rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(nn.functional.interpolate(x, scale_factor=2.0, mode="nearest"))


class Encoder(nn.Module):
    """Image -> latent moments (mean and log-variance, 2 * z_channels at 1/8 resolution)."""

    def __init__(self, in_channels: int, ch: int, ch_mult: list[int], num_res_blocks: int, z_channels: int):
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.quant_conv(self.conv_out(swish(self.norm_out(h))))


class Decoder(nn.Module):
    """Latent (z_channels at 1/8 resolution) -> image."""

    def __init__(self, ch: int, out_ch: int, ch_mult: list[int], num_res_blocks: int, z_channels: int):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def trunk(self, z: Tensor) -> Tensor:
        """Every layer except ``conv_out``: returns the ``ch``-channel features at full resolution."""
        h = self.conv_in(self.post_quant_conv(z))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        h = h.to(next(self.up.parameters()).dtype)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        return swish(self.norm_out(h))

    def forward(self, z: Tensor) -> Tensor:
        return self.conv_out(self.trunk(z))


class AutoencoderFlux2(ModelMixin, ConfigMixin):
    """FLUX.2 autoencoder: ``encode`` maps ``[B, 3, H, W]`` in [-1, 1] to ``[B, 128, H/16, W/16]``, ``decode`` inverts it.

    The latent is the posterior mean, packed 2x2 space-to-depth and normalised by a BatchNorm with fixed running
    statistics. The module is frozen and always in eval mode.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_ch: int = 3,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        z_channels: int = 32,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, ch, list(ch_mult), num_res_blocks, z_channels)
        self.decoder = Decoder(ch, out_ch, list(ch_mult), num_res_blocks, z_channels)
        self.bn = nn.BatchNorm2d(4 * z_channels, eps=1e-4, momentum=0.1, affine=False, track_running_stats=True)
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def pack(z: Tensor) -> Tensor:
        return rearrange(z, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2)

    @staticmethod
    def unpack(z: Tensor) -> Tensor:
        return rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=2, pj=2)

    def normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn.eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        return self.normalize(self.pack(torch.chunk(moments, 2, dim=1)[0]))

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.unpack(self.inv_normalize(z)))

    def train(self, mode: bool = True) -> "AutoencoderFlux2":
        return super().train(False)
