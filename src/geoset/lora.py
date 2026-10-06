"""LoRA for the GeoSET transformer: rank-r adapters on the attention and MLP projections of every block.

An adapter directory holds ``adapter_model.safetensors`` (``<module path>.lora_A`` ``[r, in]`` and
``<module path>.lora_B`` ``[out, r]``, float32) and ``adapter_config.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

LORA_TARGETS = (
    "img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
    "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2",
    "linear1", "linear2",
)
_FINGERPRINT_TENSORS = ("img_in.weight", "final_layer.linear.weight", "time_in.in_layer.weight")


class LoRALinear(nn.Linear):
    """``W x + (alpha / r) B A x`` sharing the base weight; ``A`` is Kaiming-uniform, ``B`` starts at zero."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.register_parameter("bias", base.bias)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        kw = dict(dtype=self.weight.dtype, device=self.weight.device)
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, **kw))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, **kw))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def add_lora(model: nn.Module, rank: int = 16, alpha: float = 16.0) -> list[str]:
    """Wrap the target projections in place and freeze every other parameter; returns the wrapped module paths."""
    targets = [name for name, mod in model.named_modules()
               if isinstance(mod, nn.Linear) and name.startswith(("double_blocks.", "single_blocks."))
               and name.split(".", 2)[2] in LORA_TARGETS]
    model.requires_grad_(False)
    for path in targets:
        parent, _, leaf = path.rpartition(".")
        setattr(model.get_submodule(parent), leaf, LoRALinear(model.get_submodule(path), rank, alpha))
    for p in lora_parameters(model).values():
        p.requires_grad_(True)
    return targets


def lora_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """The adapter parameters by name, in module order."""
    return {n: p for n, p in model.named_parameters() if n.endswith((".lora_A", ".lora_B"))}


def check_lora_grads(model: nn.Module) -> int:
    """Raise unless exactly the adapter parameters carry gradients; returns their count."""
    params = dict(model.named_parameters())
    adapter = lora_parameters(model)
    wrong = [n for n, p in params.items() if p.grad is not None and n not in adapter]
    dead = [n for n, p in adapter.items() if p.grad is None]
    if wrong or dead or not adapter:
        raise RuntimeError(f"LoRA gradient check failed: base tensors with grad {wrong[:3]}, adapter tensors without {dead[:3]}")
    return len(adapter)


def base_fingerprint(model: nn.Module) -> str:
    """sha256 over three base tensors that no adapter modifies."""
    sd = model.state_dict()
    h = hashlib.sha256()
    for name in _FINGERPRINT_TENSORS:
        h.update(name.encode())
        h.update(sd[name].detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_lora(save_directory: str | os.PathLike, adapter: dict[str, Tensor], rank: int, alpha: float,
              base_fingerprint: str) -> None:
    """Write ``adapter_model.safetensors`` and ``adapter_config.json``."""
    out = Path(save_directory)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().float().cpu().contiguous() for k, v in adapter.items()}
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    config = {
        "rank": int(rank),
        "alpha": float(alpha),
        "target_modules": sorted({k.rsplit(".", 1)[0] for k in tensors}),
        "base_fingerprint": base_fingerprint,
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")


def load_lora(path: str | os.PathLike) -> tuple[dict[str, Tensor], dict]:
    """Read an adapter directory: ``(tensors, config)``."""
    path = Path(path)
    return load_file(str(path / "adapter_model.safetensors")), json.loads((path / "adapter_config.json").read_text())


@torch.no_grad()
def merge_lora(model: nn.Module, path: str | os.PathLike) -> dict:
    """Fold an adapter into an unadapted model in place: ``W += (alpha / r) B A`` (float32); returns a report."""
    if getattr(model, "_lora_merged", None):
        raise RuntimeError(f"{model._lora_merged} is already merged into this model")
    tensors, cfg = load_lora(path)
    if base_fingerprint(model) != cfg["base_fingerprint"]:
        raise RuntimeError("this adapter was trained on different base weights")
    scale = float(cfg["alpha"]) / float(cfg["rank"])
    params = dict(model.named_parameters())
    num = den = 0.0
    for name in cfg["target_modules"]:
        w = params[f"{name}.weight"]
        d = (tensors[f"{name}.lora_B"].float() @ tensors[f"{name}.lora_A"].float()) * scale
        num += float(d.pow(2).sum())
        den += float(w.detach().float().pow(2).sum())
        w.add_(d.to(w.dtype, copy=False).to(w.device))
    model._lora_merged = str(path)
    return {"rank": cfg["rank"], "alpha": cfg["alpha"], "merged_modules": len(cfg["target_modules"]),
            "delta_rel_fro": math.sqrt(num / den)}
