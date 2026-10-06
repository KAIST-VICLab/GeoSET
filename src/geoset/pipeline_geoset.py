"""GeoSET SAR-to-EO translation pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from PIL import Image

from .lora import merge_lora

HUB_REPO = "JeonghyeokDo/GeoSET"
LORA_ADAPTERS = ("qxs-saropt", "sar2opt", "sar2eo", "spacenet6")


def sar_to_tensor(image: Image.Image) -> torch.Tensor:
    """8-bit SAR image -> ``[1, H, W]`` float32 in [-1, 1] (colour inputs are reduced to luminance)."""
    return torch.from_numpy(np.asarray(image.convert("L"), np.float32)[None] / 127.5 - 1.0)


class GeoSETPipeline(DiffusionPipeline):
    """SAR image -> EO image.

    Components: ``transformer`` (GeoSETTransformer2DModel), ``sar_encoder`` (GeoSETSAREncoder), ``vae``
    (AutoencoderFlux2) and ``scheduler`` (FlowBridgeScheduler). The SAR encoder maps the input to the condition
    latent ``z_s``; the flow bridge is integrated from Gaussian noise with classifier-free guidance
    ``v_u + w (v_c - v_u)`` (all-zero null condition) and the result is decoded by the FLUX.2 autoencoder. The
    transformer and the SAR encoder run in float32 (the transformer under bfloat16 autocast on CUDA); the
    autoencoder decodes in bfloat16.
    """

    model_cpu_offload_seq = "sar_encoder->transformer->vae"

    def __init__(
        self,
        transformer: ModelMixin,
        sar_encoder: ModelMixin,
        vae: ModelMixin,
        scheduler: SchedulerMixin,
    ):
        super().__init__()
        vae.to(dtype=torch.bfloat16, memory_format=torch.channels_last)
        self.register_modules(transformer=transformer, sar_encoder=sar_encoder, vae=vae, scheduler=scheduler)

    def load_lora(self, adapter: Union[str, os.PathLike], repo_id: str = HUB_REPO) -> dict:
        """Merge a dataset adapter into the transformer: a released key (``LORA_ADAPTERS``) or an adapter directory."""
        path = Path(adapter)
        if not path.is_dir():
            if adapter not in LORA_ADAPTERS:
                raise ValueError(f"{adapter!r} is neither an adapter directory nor one of {LORA_ADAPTERS}")
            root = Path(self.name_or_path).parent if self.name_or_path else None
            if root is None or not (root / "lora" / adapter).is_dir():
                root = Path(snapshot_download(repo_id, allow_patterns=[f"lora/{adapter}/*"]))
            path = root / "lora" / adapter
        return merge_lora(self.transformer, path)

    @staticmethod
    def preprocess(image: Union[Image.Image, list, torch.Tensor]) -> torch.Tensor:
        """PIL image(s) -> ``[B, 1, H, W]`` in [-1, 1]; tensors ``[C, H, W]`` / ``[B, C, H, W]`` must be in [-1, 1]."""
        if isinstance(image, torch.Tensor):
            return image if image.ndim == 4 else image[None]
        images = image if isinstance(image, (list, tuple)) else [image]
        return torch.stack([sar_to_tensor(im) for im in images])

    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, list, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,
        generator: Optional[Union[torch.Generator, list]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, tuple]:
        """Translate SAR image(s); height and width must be multiples of 16.

        Returns EO images as PIL (8-bit RGB), or as float tensors / arrays in [0, 1] (``"pt"`` ``[B, 3, H, W]``,
        ``"np"`` ``[B, H, W, 3]``).
        """
        if output_type not in ("pil", "np", "pt"):
            raise ValueError(f"output_type must be 'pil', 'np' or 'pt', got {output_type!r}")
        device = self._execution_device
        sar = self.preprocess(image).to(device=device, dtype=torch.float32)
        if sar.shape[-2] % 16 or sar.shape[-1] % 16:
            raise ValueError(f"SAR size {tuple(sar.shape[-2:])} must be a multiple of 16")
        z_s = self.sar_encoder(sar)
        if latents is None:
            latents = randn_tensor(z_s.shape, generator=generator, device=device, dtype=torch.float32)
        latents = latents.to(device=device, dtype=torch.float32)
        null = torch.zeros_like(z_s)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for t in self.progress_bar(self.scheduler.timesteps):
                t_batch = t.expand(latents.shape[0])
                v = self.transformer(latents, t_batch, z_s)
                if guidance_scale != 1.0:
                    v_u = self.transformer(latents, t_batch, null)
                    v = v_u + guidance_scale * (v - v_u)
                latents = self.scheduler.step(v, t, latents, return_dict=False)[0]
            image = self.vae.decode(latents.to(self.vae.dtype)).float()
        image = ((image + 1) / 2).clamp(0, 1)
        self.maybe_free_model_hooks()

        if output_type == "pil":
            arr = (image * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            image = [Image.fromarray(a) for a in arr]
        elif output_type == "np":
            image = image.permute(0, 2, 3, 1).cpu().numpy()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
