"""Domain-aware speckle augmentation (Eq. 2 and Appendix C of the paper).

A mean-one multiplicative field ``g ~ Gamma(L, rate=L)``, with ``L`` drawn uniformly from {16, 8, 4}
per sample, is applied in the value domain of the SAR source:

    amplitude   A * sqrt(g)
    db          x + 10 log10(g)
    display     display intensity * (1 + kappa (g - 1)),  kappa = 0.5

One field is shared by all channels of a sample. The augmentation is applied to SAR inputs during
pretraining and adaptation, never at inference.
"""

from __future__ import annotations

import math

import torch

LOOKS = (16, 8, 4)
KAPPA = 0.5
DISPLAY_CENTER, DISPLAY_SCALE = -1.0, 2.0  # [-1, 1] tensors encode display intensity in [0, 1]
DOMAINS = ("amplitude", "db", "display")


def speckle(x: torch.Tensor, domain: str, generator: torch.Generator) -> torch.Tensor:
    """Speckle one sample ``x`` ``[C, H, W]`` in ``domain``; all randomness comes from ``generator``."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown SAR value domain {domain!r}; expected one of {DOMAINS}")
    torch.rand(1, generator=generator)  # application draw (the augmentation is always applied)
    looks = float(LOOKS[int(torch.randint(len(LOOKS), (1,), generator=generator))])
    seed = int(torch.randint(0, 2**63 - 1, (1,), dtype=torch.int64, generator=generator))
    x = x[None]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        conc = torch.full((1, 1, *x.shape[2:]), looks, dtype=torch.float32)
        g = torch.distributions.Gamma(conc, looks).sample()
    g = g.expand(x.shape).contiguous()
    x32 = x.float()
    if domain == "amplitude":
        out = x32 * torch.sqrt(g)
    elif domain == "db":
        out = x32 + (10.0 / math.log(10.0)) * torch.log(g.clamp_min(1e-12))
    else:
        d = (x32 - DISPLAY_CENTER) / DISPLAY_SCALE
        d = d * (1.0 + KAPPA * (g - 1.0))
        out = d * DISPLAY_SCALE + DISPLAY_CENTER
    return out.to(x.dtype)[0]


def speckle_display(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Display-domain speckle followed by clipping to the 8-bit range [-1, 1]."""
    return speckle(x, "display", generator).clamp(-1.0, 1.0)
