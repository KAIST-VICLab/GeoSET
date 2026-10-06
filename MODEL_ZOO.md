# Model Zoo

All released weights are in the Hugging Face model repository [`JeonghyeokDo/GeoSET`](https://huggingface.co/JeonghyeokDo/GeoSET).
Sizes are decimal (1 GB = 10⁹ bytes).

## GeoSET

| Weights | Hub folder | Files | Size | Licence |
|---|---|---|---:|---|
| Generalist transformer (Stage 2, 500,000 updates, EMA) | [`generalist/transformer`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/generalist/transformer) | `diffusion_pytorch_model.safetensors`, `config.json` | 15.41 GB | CC BY-NC-SA 4.0 |
| Speckle-robust SAR encoder (Stage 1, 50,000 updates) | [`generalist/sar_encoder`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/generalist/sar_encoder) | `diffusion_pytorch_model.safetensors`, `config.json` | 137.7 MB | CC BY-NC-SA 4.0 |
| Autoencoder (FLUX.2-klein-base-4B, float32) | [`generalist/vae`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/generalist/vae) | `diffusion_pytorch_model.safetensors`, `config.json` | 336.2 MB | Apache-2.0 |
| LoRA adapter, QXS-SAROPT | [`lora/qxs-saropt`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/qxs-saropt) | `adapter_model.safetensors`, `adapter_config.json` | 92.4 MB | CC BY-NC-SA 4.0 |
| LoRA adapter, SAR2Opt | [`lora/sar2opt`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/sar2opt) | `adapter_model.safetensors`, `adapter_config.json` | 92.4 MB | CC BY-NC-SA 4.0 |
| LoRA adapter, SAR2EO | [`lora/sar2eo`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/sar2eo) | `adapter_model.safetensors`, `adapter_config.json` | 92.4 MB | CC BY-NC-SA 4.0 |
| LoRA adapter, SpaceNet6 | [`lora/spacenet6`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/spacenet6) | `adapter_model.safetensors`, `adapter_config.json` | 92.4 MB | CC BY-NC-SA 4.0 |

`generalist/` also holds `model_index.json`, the scheduler configuration and the pipeline code, so it
loads as one diffusers pipeline. Each LoRA adapter (rank 16, α 16, 20,000 updates at batch size 16) is
applied on top of the generalist. GeoSET (full FT) weights are not released; they are produced by
`scripts/train_stage2.py --full-finetune` (see the [README](README.md)).

## Results and comparison-method weights

The numbers are those of Tables 2 and 3 of the paper. All competing methods are retrained on the same
training splits and evaluated on the same test pairs ([docs/DATASETS.md](docs/DATASETS.md),
[`splits/`](splits/README.md)) with `scripts/evaluate.py`. Code, pinned upstream commits and commands for
every comparison method are in [`baselines/`](baselines/README.md).

### QXS-SAROPT · 3,999 test pairs, 256 × 256

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights | Licence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| *General image-to-image translation methods* | | | | | | | | | | |
| pix2pix | CVPR'17 | 174.6 | 0.373 | 0.1713 | 0.275 | 0.665 | 0.203 | 12.33 | [qxs-saropt/pix2pix](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/pix2pix) | BSD |
| CycleGAN | ICCV'17 | 115.5 | 0.362 | 0.0802 | 0.283 | 0.647 | 0.278 | 13.25 | [qxs-saropt/cyclegan](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/cyclegan) | BSD |
| pix2pixHD | CVPR'18 | 85.7 | 0.298 | 0.0492 | 0.403 | 0.573 | 0.358 | 16.13 | [qxs-saropt/pix2pixhd](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/pix2pixhd) | BSD |
| SPADE | CVPR'19 | 90.7 | 0.292 | 0.0607 | 0.366 | 0.599 | 0.320 | 14.53 | [qxs-saropt/spade](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/spade) | CC BY-NC-SA 4.0 |
| DDPM (SR3) | TPAMI'22 | 43.8 | 0.311 | 0.0189 | 0.425 | 0.620 | 0.359 | 14.04 | [qxs-saropt/ddpm-sr3](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/ddpm-sr3) | no upstream licence |
| SD2.1 FT | CVPR'22 | 19.1 | 0.257 | 0.0042 | 0.489 | 0.561 | 0.348 | 15.40 | [qxs-saropt/sd21-ft](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/sd21-ft) | Open RAIL++-M |
| BBDM | CVPR'23 | 76.6 | 0.270 | 0.0479 | 0.414 | 0.568 | 0.352 | 15.34 | [qxs-saropt/bbdm](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/bbdm) | MIT |
| ControlNet | ICCV'23 | 50.4 | 0.307 | 0.0211 | 0.458 | 0.604 | 0.297 | 13.42 | [qxs-saropt/controlnet](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/controlnet) | Open RAIL++-M |
| HI-Diff | NeurIPS'23 | 324.3 | 0.539 | 0.3269 | 0.215 | 0.692 | 0.457 | 17.10 | [qxs-saropt/hi-diff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/hi-diff) | Apache-2.0 |
| ResShift | NeurIPS'23 | 140.2 | 0.334 | 0.0872 | 0.295 | 0.607 | 0.217 | 14.20 | [qxs-saropt/resshift](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/resshift) | S-Lab 1.0 (non-commercial) |
| StegoGAN | CVPR'24 | 106.8 | 0.384 | 0.0707 | 0.261 | 0.658 | 0.254 | 12.96 | [qxs-saropt/stegogan](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/stegogan) | no upstream licence |
| *SAR-to-EO image translation (SET) methods* | | | | | | | | | | |
| CondDiff | GRSL'23 | 88.6 | 0.355 | 0.0537 | 0.310 | 0.730 | 0.213 | 11.55 | [qxs-saropt/conddiff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/conddiff) | no upstream licence |
| E3Diff | GRSL'24 | 47.8 | 0.278 | 0.0167 | 0.379 | 0.530 | 0.302 | 16.44 | [qxs-saropt/e3diff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/e3diff) | no upstream licence |
| cBBDM | GRSL'25 | 50.6 | 0.246 | 0.0284 | 0.492 | 0.539 | 0.372 | 16.02 | [qxs-saropt/cbbdm](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/cbbdm) | MIT |
| C-DiffSET | TCSVT'26 | 19.9 | 0.233 | 0.0055 | 0.522 | 0.526 | 0.380 | 16.92 | [qxs-saropt/c-diffset](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/qxs-saropt/c-diffset) | Open RAIL++-M |
| GeoSET (LoRA) | – | 19.3 | 0.254 | 0.0050 | 0.544 | 0.564 | 0.326 | 14.67 | [lora/qxs-saropt](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/qxs-saropt) + generalist | CC BY-NC-SA 4.0 |
| GeoSET (full FT) | – | 16.9 | 0.244 | 0.0032 | 0.558 | 0.553 | 0.332 | 15.07 | not released | – |

### SAR2Opt · 627 test pairs, center 512 × 512 crop

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights | Licence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| *General image-to-image translation methods* | | | | | | | | | | |
| pix2pix | CVPR'17 | 261.9 | 0.347 | 0.2164 | 0.277 | 0.657 | 0.199 | 13.39 | [sar2opt/pix2pix](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/pix2pix) | BSD |
| CycleGAN | ICCV'17 | 139.1 | 0.323 | 0.0343 | 0.374 | 0.642 | 0.188 | 12.68 | [sar2opt/cyclegan](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/cyclegan) | BSD |
| pix2pixHD | CVPR'18 | 146.3 | 0.283 | 0.0654 | 0.475 | 0.567 | 0.268 | 15.95 | [sar2opt/pix2pixhd](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/pix2pixhd) | BSD |
| SPADE | CVPR'19 | 142.5 | 0.265 | 0.0518 | 0.447 | 0.597 | 0.234 | 14.47 | [sar2opt/spade](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/spade) | CC BY-NC-SA 4.0 |
| DDPM (SR3) | TPAMI'22 | 122.5 | 0.295 | 0.0437 | 0.497 | 0.610 | 0.313 | 13.65 | [sar2opt/ddpm-sr3](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/ddpm-sr3) | no upstream licence |
| SD2.1 FT | CVPR'22 | 71.8 | 0.211 | 0.0094 | 0.600 | 0.541 | 0.293 | 16.24 | [sar2opt/sd21-ft](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/sd21-ft) | Open RAIL++-M |
| BBDM | CVPR'23 | 143.1 | 0.290 | 0.0671 | 0.466 | 0.590 | 0.276 | 15.29 | [sar2opt/bbdm](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/bbdm) | MIT |
| ControlNet | ICCV'23 | 140.5 | 0.350 | 0.0480 | 0.479 | 0.643 | 0.217 | 11.73 | [sar2opt/controlnet](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/controlnet) | Open RAIL++-M |
| HI-Diff | NeurIPS'23 | 319.8 | 0.473 | 0.2357 | 0.277 | 0.692 | 0.384 | 17.36 | [sar2opt/hi-diff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/hi-diff) | Apache-2.0 |
| ResShift | NeurIPS'23 | 141.7 | 0.304 | 0.0515 | 0.435 | 0.597 | 0.177 | 14.31 | [sar2opt/resshift](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/resshift) | S-Lab 1.0 (non-commercial) |
| StegoGAN | CVPR'24 | 149.8 | 0.332 | 0.0396 | 0.362 | 0.652 | 0.162 | 12.39 | [sar2opt/stegogan](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/stegogan) | no upstream licence |
| *SAR-to-EO image translation (SET) methods* | | | | | | | | | | |
| CondDiff | GRSL'23 | 211.8 | 0.415 | 0.1379 | 0.343 | 0.686 | 0.248 | 12.48 | [sar2opt/conddiff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/conddiff) | no upstream licence |
| E3Diff | GRSL'24 | 104.7 | 0.232 | 0.0306 | 0.541 | 0.529 | 0.249 | 16.09 | [sar2opt/e3diff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/e3diff) | no upstream licence |
| cBBDM | GRSL'25 | 222.3 | 0.377 | 0.1521 | 0.413 | 0.571 | 0.361 | 17.05 | [sar2opt/cbbdm](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/cbbdm) | MIT |
| C-DiffSET | TCSVT'26 | 78.1 | 0.214 | 0.0138 | 0.601 | 0.529 | 0.314 | 16.81 | [sar2opt/c-diffset](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2opt/c-diffset) | Open RAIL++-M |
| GeoSET (LoRA) | – | 74.6 | 0.200 | 0.0066 | 0.626 | 0.535 | 0.278 | 15.50 | [lora/sar2opt](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/sar2opt) + generalist | CC BY-NC-SA 4.0 |
| GeoSET (full FT) | – | 71.1 | 0.196 | 0.0052 | 0.614 | 0.532 | 0.279 | 15.66 | not released | – |

### SAR2EO · first 4,000 of the 21,260 test pairs, 256 × 256

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights | Licence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| *General image-to-image translation methods* | | | | | | | | | | |
| pix2pix | CVPR'17 | 169.9 | 0.315 | 0.1551 | 0.400 | 0.573 | 0.455 | 18.06 | [sar2eo/pix2pix](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/pix2pix) | BSD |
| CycleGAN | ICCV'17 | 360.7 | 0.522 | 0.3903 | 0.369 | 0.683 | 0.300 | 15.17 | [sar2eo/cyclegan](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/cyclegan) | BSD |
| pix2pixHD | CVPR'18 | 106.0 | 0.254 | 0.0638 | 0.543 | 0.439 | 0.619 | 21.68 | [sar2eo/pix2pixhd](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/pix2pixhd) | BSD |
| SPADE | CVPR'19 | 98.5 | 0.260 | 0.0646 | 0.504 | 0.501 | 0.562 | 19.81 | [sar2eo/spade](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/spade) | CC BY-NC-SA 4.0 |
| DDPM (SR3) | TPAMI'22 | 85.0 | 0.272 | 0.0390 | 0.529 | 0.471 | 0.479 | 13.49 | [sar2eo/ddpm-sr3](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/ddpm-sr3) | no upstream licence |
| SD2.1 FT | CVPR'22 | 67.7 | 0.270 | 0.0282 | 0.511 | 0.487 | 0.459 | 15.46 | [sar2eo/sd21-ft](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/sd21-ft) | Open RAIL++-M |
| BBDM | CVPR'23 | 63.7 | 0.231 | 0.0245 | 0.613 | 0.418 | 0.601 | 19.08 | [sar2eo/bbdm](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/bbdm) | MIT |
| ControlNet | ICCV'23 | 87.8 | 0.315 | 0.0424 | 0.509 | 0.523 | 0.418 | 12.93 | [sar2eo/controlnet](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/controlnet) | Open RAIL++-M |
| HI-Diff | NeurIPS'23 | 315.5 | 0.556 | 0.3007 | 0.155 | 0.647 | 0.672 | 21.68 | [sar2eo/hi-diff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/hi-diff) | Apache-2.0 |
| ResShift | NeurIPS'23 | 130.6 | 0.273 | 0.0768 | 0.418 | 0.510 | 0.504 | 19.29 | [sar2eo/resshift](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/resshift) | S-Lab 1.0 (non-commercial) |
| StegoGAN | CVPR'24 | 384.7 | 0.550 | 0.4343 | 0.387 | 0.698 | 0.180 | 12.49 | [sar2eo/stegogan](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/stegogan) | no upstream licence |
| *SAR-to-EO image translation (SET) methods* | | | | | | | | | | |
| CondDiff | GRSL'23 | 113.2 | 0.317 | 0.0639 | 0.461 | 0.551 | 0.334 | 11.12 | [sar2eo/conddiff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/conddiff) | no upstream licence |
| E3Diff | GRSL'24 | 55.5 | 0.228 | 0.0148 | 0.550 | 0.446 | 0.514 | 20.58 | [sar2eo/e3diff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/e3diff) | no upstream licence |
| cBBDM | GRSL'25 | 52.7 | 0.191 | 0.0214 | 0.675 | 0.369 | 0.650 | 21.73 | [sar2eo/cbbdm](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/cbbdm) | MIT |
| C-DiffSET | TCSVT'26 | 64.0 | 0.252 | 0.0257 | 0.527 | 0.462 | 0.512 | 17.09 | [sar2eo/c-diffset](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/sar2eo/c-diffset) | Open RAIL++-M |
| GeoSET (LoRA) | – | 29.5 | 0.202 | 0.0042 | 0.691 | 0.440 | 0.547 | 19.19 | [lora/sar2eo](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/sar2eo) + generalist | CC BY-NC-SA 4.0 |
| GeoSET (full FT) | – | 24.5 | 0.181 | 0.0032 | 0.730 | 0.399 | 0.564 | 19.72 | not released | – |

### SpaceNet6 · 495 test pairs, 256 × 256

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights | Licence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| *General image-to-image translation methods* | | | | | | | | | | |
| pix2pix | CVPR'17 | 166.7 | 0.242 | 0.1091 | 0.589 | 0.419 | 0.459 | 17.07 | [spacenet6/pix2pix](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/pix2pix) | BSD |
| CycleGAN | ICCV'17 | 119.3 | 0.219 | 0.0436 | 0.660 | 0.373 | 0.478 | 16.88 | [spacenet6/cyclegan](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/cyclegan) | BSD |
| pix2pixHD | CVPR'18 | 175.1 | 0.246 | 0.1136 | 0.682 | 0.361 | 0.525 | 19.21 | [spacenet6/pix2pixhd](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/pix2pixhd) | BSD |
| SPADE | CVPR'19 | 179.7 | 0.242 | 0.1228 | 0.662 | 0.385 | 0.489 | 17.77 | [spacenet6/spade](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/spade) | CC BY-NC-SA 4.0 |
| DDPM (SR3) | TPAMI'22 | 219.1 | 0.333 | 0.1616 | 0.419 | 0.622 | 0.111 | 12.45 | [spacenet6/ddpm-sr3](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/ddpm-sr3) | no upstream licence |
| SD2.1 FT | CVPR'22 | 124.7 | 0.196 | 0.0537 | 0.612 | 0.348 | 0.513 | 19.29 | [spacenet6/sd21-ft](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/sd21-ft) | Open RAIL++-M |
| BBDM | CVPR'23 | 230.2 | 0.381 | 0.1647 | 0.470 | 0.468 | 0.449 | 18.03 | [spacenet6/bbdm](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/bbdm) | MIT |
| ControlNet | ICCV'23 | 131.3 | 0.279 | 0.0635 | 0.615 | 0.413 | 0.448 | 14.45 | [spacenet6/controlnet](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/controlnet) | Open RAIL++-M |
| HI-Diff | NeurIPS'23 | 261.2 | 0.300 | 0.2036 | 0.499 | 0.403 | 0.589 | 20.61 | [spacenet6/hi-diff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/hi-diff) | Apache-2.0 |
| ResShift | NeurIPS'23 | 155.4 | 0.230 | 0.0858 | 0.540 | 0.367 | 0.421 | 18.58 | [spacenet6/resshift](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/resshift) | S-Lab 1.0 (non-commercial) |
| StegoGAN | CVPR'24 | 113.1 | 0.225 | 0.0361 | 0.649 | 0.388 | 0.457 | 16.04 | [spacenet6/stegogan](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/stegogan) | no upstream licence |
| *SAR-to-EO image translation (SET) methods* | | | | | | | | | | |
| CondDiff | GRSL'23 | 160.1 | 0.274 | 0.1012 | 0.514 | 0.517 | 0.198 | 14.68 | [spacenet6/conddiff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/conddiff) | no upstream licence |
| E3Diff | GRSL'24 | 110.6 | 0.197 | 0.0440 | 0.696 | 0.361 | 0.462 | 18.32 | [spacenet6/e3diff](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/e3diff) | no upstream licence |
| cBBDM | GRSL'25 | 280.4 | 0.347 | 0.2620 | 0.450 | 0.419 | 0.447 | 19.72 | [spacenet6/cbbdm](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/cbbdm) | MIT |
| Seg-CycleGAN | GRSL'25 | 131.4 | 0.229 | 0.0611 | 0.649 | 0.386 | 0.471 | 16.62 | [spacenet6/seg-cyclegan](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/seg-cyclegan) | no upstream licence |
| C-DiffSET | TCSVT'26 | 132.3 | 0.197 | 0.0614 | 0.611 | 0.346 | 0.520 | 19.43 | [spacenet6/c-diffset](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/spacenet6/c-diffset) | Open RAIL++-M |
| GeoSET (LoRA) | – | 87.9 | 0.189 | 0.0162 | 0.733 | 0.338 | 0.513 | 17.95 | [lora/spacenet6](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/spacenet6) + generalist | CC BY-NC-SA 4.0 |
| GeoSET (full FT) | – | 94.1 | 0.192 | 0.0183 | 0.724 | 0.342 | 0.515 | 17.99 | not released | – |

## Licences

- **GeoSET** weights (generalist transformer, SAR encoder, LoRA adapters): CC BY-NC-SA 4.0, see
  [LICENSE-WEIGHTS.md](LICENSE-WEIGHTS.md). The bundled autoencoder is the FLUX.2-klein-base-4B
  autoencoder, Apache-2.0.
- **Comparison methods**: each checkpoint carries the terms of the code it was trained with; the texts are
  in [`baselines/licenses/`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines/licenses) on the Hub.
  - BSD: pix2pix and CycleGAN ([text](https://huggingface.co/JeonghyeokDo/GeoSET/blob/main/baselines/licenses/LICENSE-pix2pix.txt)), pix2pixHD ([text](https://huggingface.co/JeonghyeokDo/GeoSET/blob/main/baselines/licenses/LICENSE-pix2pixhd.txt)).
  - CC BY-NC-SA 4.0: SPADE ([text](https://huggingface.co/JeonghyeokDo/GeoSET/blob/main/baselines/licenses/LICENSE-spade.txt)).
  - MIT: BBDM ([text](https://huggingface.co/JeonghyeokDo/GeoSET/blob/main/baselines/licenses/LICENSE-bbdm.txt)) and cBBDM ([text](https://huggingface.co/JeonghyeokDo/GeoSET/blob/main/baselines/licenses/LICENSE-cbbdm.txt)); both checkpoints
    contain the CompVis latent-diffusion VQ-f4 autoencoder (MIT, [text](https://huggingface.co/JeonghyeokDo/GeoSET/blob/main/baselines/licenses/LICENSE-latent-diffusion.txt)).
  - Apache-2.0: HI-Diff ([text](https://huggingface.co/JeonghyeokDo/GeoSET/blob/main/baselines/licenses/LICENSE-hi-diff.txt)).
  - S-Lab License 1.0, non-commercial use only: ResShift ([text](https://huggingface.co/JeonghyeokDo/GeoSET/blob/main/baselines/licenses/LICENSE-resshift.txt)).
  - CreativeML Open RAIL++-M: SD2.1 FT, ControlNet and C-DiffSET, which are derivatives of Stable
    Diffusion 2.1-base ([text](https://huggingface.co/JeonghyeokDo/GeoSET/blob/main/baselines/licenses/LICENSE-CreativeML-OpenRAIL++-M.txt)); the use-based restrictions of
    its Attachment A apply. Their training code is MIT (C-DiffSET) or Apache-2.0 (diffusers).
  - No upstream licence: DDPM (SR3), StegoGAN, CondDiff, E3Diff and Seg-CycleGAN were trained with
    upstream code that publishes no licence ([note](https://huggingface.co/JeonghyeokDo/GeoSET/blob/main/baselines/licenses/NO-UPSTREAM-LICENSE.md)).
- **Datasets** are not redistributed; their terms are listed in [docs/DATASETS.md](docs/DATASETS.md).
