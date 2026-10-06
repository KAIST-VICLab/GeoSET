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
"""GeoSET transformer: the FLUX.2 double/single-stream transformer with its text stream replaced by a SAR stream.

The SAR stream occupies the second-stream slots of the double blocks (the ``txt_*`` modules keep their FLUX.2
names) and is concatenated with the EO tokens in the single blocks; only the EO tokens reach the final velocity head.
"""

from __future__ import annotations

import math
import os

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Full bidirectional attention: ``[B, H, L, D]`` -> ``[B, L, H * D]``."""
    x = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
    return rearrange(x, "b h n d -> b n (h d)")


def position_ids(gh: int, gw: int, batch: int, device) -> Tensor:
    """FLUX.2 ``(t, h, w, l)`` image-token ids; EO and SAR tokens share them."""
    zero = torch.zeros(1, device=device, dtype=torch.float32)
    h = torch.arange(gh, device=device, dtype=torch.float32)
    w = torch.arange(gw, device=device, dtype=torch.float32)
    return torch.cartesian_prod(zero, h, w, zero)[None].expand(batch, -1, -1)


class EmbedND(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat([rope(ids[..., i], d, self.theta) for i, d in enumerate(self.axes_dim)], dim=-3)
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class SiLUActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim, bias=False)


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=False)

    def forward(self, vec: Tensor) -> tuple[Tensor, ...]:
        return self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear((1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :])


class DoubleStreamBlock(nn.Module):
    """Separate EO (``img_*``) and SAR (``txt_*``) weights, joint attention over both token sets."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

    def _qkv(self, attn: SelfAttention, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v = rearrange(attn.qkv(x), "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        return q, k, v

    def forward(self, img: Tensor, sar: Tensor, pe: Tensor, mod_img, mod_sar) -> tuple[Tensor, Tensor]:
        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = mod_img
        sar_shift1, sar_scale1, sar_gate1, sar_shift2, sar_scale2, sar_gate2 = mod_sar
        img_q, img_k, img_v = self._qkv(self.img_attn, (1 + img_scale1) * self.img_norm1(img) + img_shift1)
        sar_q, sar_k, sar_v = self._qkv(self.txt_attn, (1 + sar_scale1) * self.txt_norm1(sar) + sar_shift1)
        q, k = apply_rope(torch.cat((sar_q, img_q), dim=2), torch.cat((sar_k, img_k), dim=2), pe)
        attn = attention(q, k, torch.cat((sar_v, img_v), dim=2))
        n = sar_q.shape[2]
        img = img + img_gate1 * self.img_attn.proj(attn[:, n:])
        img = img + img_gate2 * self.img_mlp((1 + img_scale2) * self.img_norm2(img) + img_shift2)
        sar = sar + sar_gate1 * self.txt_attn.proj(attn[:, :n])
        sar = sar + sar_gate2 * self.txt_mlp((1 + sar_scale2) * self.txt_norm2(sar) + sar_shift2)
        return img, sar


class SingleStreamBlock(nn.Module):
    """Fused attention + MLP block over the concatenated ``[SAR, EO]`` tokens."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * 2, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    def forward(self, x: Tensor, pe: Tensor, mod) -> Tensor:
        shift, scale, gate = mod
        qkv, mlp = torch.split(
            self.linear1((1 + scale) * self.pre_norm(x) + shift),
            [3 * self.hidden_size, self.mlp_hidden_dim * 2],
            dim=-1,
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return x + gate * self.linear2(torch.cat((attention(q, k, v), self.mlp_act(mlp)), 2))


class GeoSETTransformer2DModel(ModelMixin, ConfigMixin):
    """Velocity network ``v = model(z_t, t, z_s)`` on packed FLUX.2 latents ``[B, 128, h, w]``.

    ``t`` in [0, 1] is the flow time (0 = noise, 1 = EO data). The defaults are the FLUX.2-klein-base-4B geometry.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        hidden_size: int = 3072,
        num_heads: int = 24,
        depth: int = 5,
        depth_single_blocks: int = 20,
        axes_dim: tuple[int, ...] = (32, 32, 32, 32),
        theta: int = 2000,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()
        if sum(axes_dim) != hidden_size // num_heads:
            raise ValueError(f"sum(axes_dim)={sum(axes_dim)} must equal hidden_size // num_heads")
        self.img_in = nn.Linear(in_channels, hidden_size, bias=False)
        self.sar_in = nn.Linear(in_channels, hidden_size, bias=False)
        self.time_in = MLPEmbedder(256, hidden_size)
        self.pe_embedder = EmbedND(theta, list(axes_dim))
        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth_single_blocks)]
        )
        self.double_stream_modulation_img = Modulation(hidden_size, double=True)
        self.double_stream_modulation_txt = Modulation(hidden_size, double=True)
        self.single_stream_modulation = Modulation(hidden_size, double=False)
        self.final_layer = LastLayer(hidden_size, in_channels)
        self._pe_cache: dict = {}

    def _pe(self, gh: int, gw: int, batch: int, device) -> Tensor:
        key = (gh, gw, batch, str(device))
        if key not in self._pe_cache:
            self._pe_cache[key] = self.pe_embedder(position_ids(gh, gw, batch, device))
        return self._pe_cache[key]

    def forward(self, z_t: Tensor, t: Tensor, z_s: Tensor) -> Tensor:
        b, _, gh, gw = z_t.shape
        # FLUX.2 is trained with sigma = 1 at noise and predicts (noise - data): feed 1 - t and negate the output.
        vec = self.time_in(timestep_embedding(1.0 - t, 256))
        pe = self._pe(gh, gw, b, z_t.device)
        pe = torch.cat((pe, pe), dim=2)
        img = self.img_in(rearrange(z_t, "b c h w -> b (h w) c"))
        sar = self.sar_in(rearrange(z_s, "b c h w -> b (h w) c"))
        mod_img = self.double_stream_modulation_img(vec)
        mod_sar = self.double_stream_modulation_txt(vec)
        for block in self.double_blocks:
            img, sar = block(img, sar, pe, mod_img, mod_sar)
        n = sar.shape[1]
        x = torch.cat((sar, img), dim=1)
        mod = self.single_stream_modulation(vec)
        for block in self.single_blocks:
            x = block(x, pe, mod)
        return -rearrange(self.final_layer(x[:, n:], vec), "b (h w) c -> b c h w", h=gh, w=gw)


@torch.no_grad()
def load_flux2_base(model: GeoSETTransformer2DModel, path: str | os.PathLike) -> dict:
    """Stage-2 initialisation from the FLUX.2-klein-base-4B transformer weights (``flux-2-klein-base-4b.safetensors``).

    The image stream and all shared weights are loaded, the text stream is skipped, and the SAR stream is a copy of
    the image stream. Returns a load report; raises if any parameter is left uninitialised.
    """
    state = load_file(os.fspath(path))
    own = model.state_dict()
    sar = {k for k in own if k.startswith(("sar_in.", "double_stream_modulation_txt.")) or ".txt_" in k}
    loaded = {k: v for k, v in state.items() if k in own and k not in sar and own[k].shape == v.shape}
    model.load_state_dict(loaded, strict=False)
    params = dict(model.named_parameters())
    pairs = {"sar_in.weight": "img_in.weight",
             "double_stream_modulation_txt.lin.weight": "double_stream_modulation_img.lin.weight"}
    pairs.update({n.replace(".img_", ".txt_"): n for n in params if ".img_" in n})
    for dst, src in pairs.items():
        params[dst].copy_(params[src])
    missing = [k for k in own if k not in loaded and k not in pairs]
    if missing:
        raise RuntimeError(f"parameters neither loaded nor cloned: {missing}")
    n_ckpt = sum(v.numel() for v in state.values())
    n_loaded = sum(v.numel() for v in loaded.values())
    return {
        "loaded_tensors": len(loaded),
        "loaded_params": n_loaded,
        "checkpoint_params": n_ckpt,
        "loaded_fraction": n_loaded / n_ckpt,
        "cloned_params": sum(own[k].numel() for k in pairs),
        "skipped": sorted(set(state) - set(loaded)),
    }
