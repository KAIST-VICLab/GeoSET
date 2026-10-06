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
"""Speckle-robust SAR encoder (stage 1): the FLUX.2 encoder trunk with channel-specific input and output stems."""

from __future__ import annotations

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn

from .autoencoder_flux2 import AutoencoderFlux2, Encoder

_CHUNK = 32


class GeoSETSAREncoder(ModelMixin, ConfigMixin):
    """SAR image ``[B, 1|2, H, W]`` in [-1, 1] -> SAR latent ``z_s`` ``[B, 128, H/16, W/16]``.

    Stems are keyed by input channels: ``"3"`` takes single-channel SAR folded to three channels, ``"2"`` takes
    VV/VH. The latent is normalised with the autoencoder's fixed BatchNorm statistics. ``stem_out`` (stage-1
    reconstruction heads) is not used by ``forward``.
    """

    @register_to_config
    def __init__(
        self,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        z_channels: int = 32,
    ):
        super().__init__()
        self.encoder = Encoder(3, ch, list(ch_mult), num_res_blocks, z_channels)
        self.encoder.conv_in = nn.Identity()
        self.bn = nn.BatchNorm2d(4 * z_channels, eps=1e-4, momentum=0.1, affine=False, track_running_stats=True)
        self.bn.eval()
        self.stem_in = nn.ModuleDict({"3": nn.Conv2d(3, ch, 3, 1, 1), "2": nn.Conv2d(2, ch, 3, 1, 1)})
        self.stem_out = nn.ModuleDict({"3": nn.Conv2d(ch, 3, 3, 1, 1), "2": nn.Conv2d(ch, 2, 3, 1, 1)})

    def train(self, mode: bool = True) -> "GeoSETSAREncoder":
        super().train(mode)
        self.bn.eval()
        return self

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``[B, 3|2, H, W]`` -> ``(z, z_raw)``: the BatchNorm-normalised and the raw packed posterior mean."""
        moments = self.encoder(self.stem_in[str(x.shape[1])](x))
        z_raw = AutoencoderFlux2.pack(torch.chunk(moments, 2, dim=1)[0])
        return self.bn(z_raw), z_raw

    def forward(self, sar: Tensor) -> Tensor:
        if sar.shape[1] == 1:
            sar = sar.repeat(1, 3, 1, 1)
        return torch.cat([self.encode(c)[0] for c in sar.split(_CHUNK)], dim=0).float()

    def reconstruct(self, x: Tensor, vae: AutoencoderFlux2) -> Tensor:
        """Stage-1 reconstruction ``[B, 3|2, H, W]`` -> ``[B, 3|2, H, W]`` through the frozen decoder trunk."""
        _, z_raw = self.encode(x)
        return self.stem_out[str(x.shape[1])](vae.decoder.trunk(AutoencoderFlux2.unpack(z_raw)))

    @classmethod
    @torch.no_grad()
    def from_autoencoder(cls, vae: AutoencoderFlux2) -> "GeoSETSAREncoder":
        """Stage-1 initialisation from the FLUX.2 autoencoder.

        The trunk and latent BatchNorm are copies of the encoder's; ``conv_in`` and ``conv_out`` become the
        three-channel stems; the two-channel stems start from their input-channel sum and output-channel mean.
        Trainable: the trunk, both input stems and ``stem_out["2"]``; ``stem_out["3"]`` and the BatchNorm stay fixed.
        """
        c = vae.config
        model = cls(ch=c.ch, ch_mult=c.ch_mult, num_res_blocks=c.num_res_blocks, z_channels=c.z_channels)
        model.encoder.load_state_dict({k: v for k, v in vae.encoder.state_dict().items()
                                       if not k.startswith("conv_in.")})
        model.bn.load_state_dict(vae.bn.state_dict())
        w_in, b_in = vae.encoder.conv_in.weight.float(), vae.encoder.conv_in.bias.float()
        w_out, b_out = vae.decoder.conv_out.weight.float(), vae.decoder.conv_out.bias.float()
        stems = {
            ("stem_in", "3"): (w_in, b_in),
            ("stem_in", "2"): (w_in.sum(dim=1, keepdim=True).repeat(1, 2, 1, 1) / 2, b_in.clone()),
            ("stem_out", "3"): (w_out, b_out),
            ("stem_out", "2"): (w_out.mean(dim=0, keepdim=True).repeat(2, 1, 1, 1), b_out.mean().repeat(2)),
        }
        for (group, key), (w, b) in stems.items():
            getattr(model, group)[key].weight.copy_(w)
            getattr(model, group)[key].bias.copy_(b)
        model.stem_out["3"].requires_grad_(False)
        return model
