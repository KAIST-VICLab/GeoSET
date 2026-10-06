<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.svg">
    <img src="assets/logo.svg" alt="GeoSET" width="440">
  </picture>
</p>

<div align="center">
<h2>GeoSET: Generalist Foundation Model for SAR-to-EO Image Translation</h2>

<div>
    <a href='https://jeonghyeokdo.github.io/' target='_blank'>Jeonghyeok Do</a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href='https://www.viclab.kaist.ac.kr/' target='_blank'>Munchurl Kim</a><sup>†</sup>
</div>
<div>
    Korea Advanced Institute of Science and Technology (KAIST), South Korea
</div>
<div>
    <sup>†</sup>Corresponding author
</div>

<div>
    <h4 align="center">
        <a href="https://kaist-viclab.github.io/GeoSET_site/" target='_blank'>
        <img src="https://img.shields.io/badge/🏠-Project%20Page-blue">
        </a>
        <!-- ARXIV_BADGE_START --><a href="https://arxiv.org/abs/2609.37496" target="_blank"><img src="https://img.shields.io/badge/arXiv-2609.37496-b31b1b.svg" alt="arXiv"></a><!-- ARXIV_BADGE_END -->
        <a href="https://huggingface.co/JeonghyeokDo/GeoSET" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GeoCR-yellow"></a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KAIST-VICLab/GeoSET">
    </h4>
</div>
</div>

---

<div align="center">
    <h4>
        This is the official repository of "GeoSET: Generalist Foundation Model for SAR-to-EO Image Translation".
    </h4>
</div>

## 📧 News
- **Oct 2026:** The code, the pretrained generalist, the LoRA adapters for the four public benchmarks and the retrained comparison methods are released.
- **Sep 2026:** This repository is created. The code will be released soon.

## 📖 Abstract
Paired synthetic aperture radar (SAR) and electro-optical (EO) imagery is increasingly available across sensors, resolutions, and geographic regions. Yet existing SAR-to-EO image translation (SET) methods are typically trained on a single, limited-scale dataset, producing models specialized to particular sensing conditions. We introduce **GeoSET**, *the first generalist model for SET*, built around a single pretrained parent that is adapted to downstream datasets under a common protocol. We curate over 3 million high-quality SAR–EO pairs from a collection of more than 10 million SAR observations, spanning diverse sensors, spatial resolutions, and ground sampling distances. To bridge the modality gap between SAR observations and a pretrained image generator, we develop a speckle-robust SAR encoder and pretrain the conditional generator on this heterogeneous corpus. The resulting parent supports efficient adaptation across downstream datasets through low-rank adaptation (LoRA), updating only 0.60% of the generator parameters and requiring approximately one hour per dataset. Across six downstream benchmarks, GeoSET achieves state-of-the-art results in FID and DISTS with full fine-tuning or LoRA, demonstrating effective transfer across heterogeneous SAR–EO domains.

## 📊 Results
Across full fine-tuning and LoRA, GeoSET achieves the best reported FID on all six downstream benchmarks and the best DISTS on five.

### Qualitative Comparison
Columns (g)–(h): GeoSET with LoRA and full fine-tuning; (i): ground-truth EO.

<div align="center">
    <img src="assets/teaser.jpg" alt="Qualitative comparison on SAR-to-EO image translation benchmarks" width="100%">
</div>

### Cross-Dataset Comparison
FID and DISTS on six benchmarks, normalized for each dataset–metric pair as 100 × best / value (outer ring = best).

<div align="center">
    <img src="assets/radar.png" alt="Cross-dataset comparison of SAR-to-EO image translation methods" width="65%">
</div>

### Quantitative Comparison
All competing methods are retrained and evaluated on the same splits. FID and DISTS are the primary metrics; LPIPS, PSNR and SSIM are retained as complementary fidelity measures.

<div align="center">
    <img src="assets/table_qxs_sar2opt.png" alt="Quantitative comparison on QXS-SAROPT and SAR2Opt" width="100%">
</div>
<br>
<div align="center">
    <img src="assets/table_sar2eo_spacenet6.png" alt="Quantitative comparison on SAR2EO and SpaceNet6" width="100%">
</div>

**Please visit our [project page](https://kaist-viclab.github.io/GeoSET_site/) for the interactive gallery and more results.**

## 🖼️ Method Overview

<div align="center">
    <img src="assets/framework.jpg" alt="Overview of the GeoSET framework" width="100%">
</div>

- **Stage 1 · Speckle-robust SAR encoder:** reconstructs the original SAR observation from a speckle-perturbed copy through a frozen decoder.
- **Stage 2 · Generalist pretraining:** a SAR-conditioned FLUX.2 flow transformer is trained on 3,204,744 curated SAR–EO pairs.
- **Stage 3 · Downstream adaptation:** the same parent is adapted to each benchmark by LoRA (0.60% of generator parameters, about one hour per dataset) or full fine-tuning.

## ⚙️ Installation

```bash
git clone https://github.com/KAIST-VICLab/GeoSET.git
cd GeoSET

conda create -n geoset python=3.12 -y
conda activate geoset

# Install the torch/torchvision build matching your CUDA setup first (see https://pytorch.org),
# then the remaining dependencies:
pip install -r requirements.txt
```

The scripts in `scripts/` import the `geoset` package from `src/` themselves. To import it in your own code,
add `src/` to `PYTHONPATH`.

## 📦 Pretrained Models

All weights are in the Hugging Face model repository [`JeonghyeokDo/GeoSET`](https://huggingface.co/JeonghyeokDo/GeoSET).

| Weights | Hub folder | Size | Licence |
|---|---|---:|---|
| Generalist transformer (Stage 2, 500,000 updates, EMA) | [`generalist/transformer`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/generalist/transformer) | 15.41 GB | CC BY-NC-SA 4.0 |
| Speckle-robust SAR encoder (Stage 1, 50,000 updates) | [`generalist/sar_encoder`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/generalist/sar_encoder) | 137.7 MB | CC BY-NC-SA 4.0 |
| Autoencoder (FLUX.2-klein-base-4B, float32) | [`generalist/vae`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/generalist/vae) | 336.2 MB | Apache-2.0 |
| LoRA adapter, QXS-SAROPT | [`lora/qxs-saropt`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/qxs-saropt) | 92.4 MB | CC BY-NC-SA 4.0 |
| LoRA adapter, SAR2Opt | [`lora/sar2opt`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/sar2opt) | 92.4 MB | CC BY-NC-SA 4.0 |
| LoRA adapter, SAR2EO | [`lora/sar2eo`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/sar2eo) | 92.4 MB | CC BY-NC-SA 4.0 |
| LoRA adapter, SpaceNet6 | [`lora/spacenet6`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/lora/spacenet6) | 92.4 MB | CC BY-NC-SA 4.0 |

`generalist/` is a complete `diffusers` pipeline (`model_index.json`, the three models, the scheduler and the
pipeline code). Each LoRA adapter (rank 16, α 16, 20,000 updates at batch size 16) is applied on top of the
generalist. The full fine-tuned models are not released; they are produced by Stage 3 with `--full-finetune`
(see Training). Metrics of every released checkpoint are in [MODEL_ZOO.md](MODEL_ZOO.md).

To keep a local copy:

```bash
hf download JeonghyeokDo/GeoSET --include "generalist/*" --include "lora/*" --local-dir weights/GeoSET
```

## ⚡ Inference

With `diffusers` only (no checkout of this repository needed):

```python
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download

root = snapshot_download("JeonghyeokDo/GeoSET", allow_patterns=["generalist/*", "lora/sar2opt/*"])
pipe = DiffusionPipeline.from_pretrained(f"{root}/generalist", custom_pipeline=f"{root}/generalist",
                                         torch_dtype=torch.float32).to("cuda")
pipe.load_lora("sar2opt")                                   # optional; resolved from {root}/lora/sar2opt
eo = pipe(Image.open("sar.png"), num_inference_steps=50, guidance_scale=2.0,
          generator=torch.Generator("cuda").manual_seed(20260812)).images[0]
eo.save("eo.png")
```

`custom_pipeline` points at the same folder because the pipeline, model and scheduler classes ship with the
checkpoint. Keep `torch_dtype=torch.float32`; the pipeline casts the autoencoder to bfloat16 itself. The
input is an 8-bit SAR image, read as one channel, or a list of equally sized images; every side must be a
multiple of 16. Images are not resized or cropped inside the pipeline, so SAR2Opt tiles (600 × 600) are
center-cropped to 512 × 512 first. The paper's results use **50 integration steps with two-pass classifier-free guidance of weight
2.0**.

**LoRA adapters.** `pipe.load_lora(name)` merges one adapter into the transformer weights: `qxs-saropt`,
`sar2opt`, `sar2eo` or `spacenet6`, or the path of an adapter directory written by Stage 3. Without it, the
pipeline runs the pretrained generalist. An adapter is merged once; load the pipeline again to switch to
another one.

**Folders and benchmark test splits** with `scripts/translate.py`:

```bash
# your own SAR images (every side a multiple of 16)
python scripts/translate.py --input-dir path/to/sar --out-dir outputs/eo

# a benchmark test split with the pretrained generalist
python scripts/translate.py --dataset qxs-saropt --data-root $DATA_ROOT --out-dir outputs/geoset/qxs-saropt

# a benchmark test split with its LoRA adapter
python scripts/translate.py --dataset sar2opt --data-root $DATA_ROOT --lora sar2opt --out-dir outputs/geoset-lora/sar2opt
```

The defaults are the paper's settings (`--steps 50 --guidance 2.0`). `--dataset` reads the frozen test list
in `splits/<dataset>/` (for `sar2eo`, the first 4,000 test pairs), center-crops SAR2Opt to 512 × 512 and
writes `<out-dir>/<stem>.png`; batch `b` starts from noise seeded with `--seed` + `b`, and batches whose
outputs exist are skipped, so an interrupted run resumes with the same command. Weights are fetched from the
Hub by default; `--checkpoint weights/GeoSET` uses the local copy above. Your own Stage 3 results run with
`--lora <adapter dir>` or `--transformer <fine-tuned transformer dir>`.

## 🗂️ Data Preparation

**Downstream benchmarks.** [docs/DATASETS.md](docs/DATASETS.md) gives the source, licence, download and
preparation commands and checksums of the four public benchmarks, arranged under one directory `$DATA_ROOT`:

| Dataset | Key | Train / test | Evaluation size |
|---|---|---:|---|
| QXS-SAROPT | `qxs-saropt` | 16,001 / 3,999 | 256 × 256 |
| SAR2Opt | `sar2opt` | 1,450 / 627 | center 512 × 512 crop of the 600 × 600 tiles |
| SAR2EO | `sar2eo` | 68,151 / 4,000 | 256 × 256 (first 4,000 of the 21,260 test pairs) |
| SpaceNet6 | `spacenet6` | 2,558 / 495 | 256 × 256 |

The frozen train/test lists used for every result in the paper are in [`splits/`](splits/README.md).
SpaceNet6 is read as 256-pixel chips rendered once from the GeoTIFFs of the official training archive:

```bash
python scripts/datasets/build_spacenet6.py --data-root $DATA_ROOT --verify-stretch
```

**Pretraining corpus.** GeoSET is pretrained on GUSO, TerraMesh, SARLO-80, SAR-1M and 3MOS.
[docs/CORPUS.md](docs/CORPUS.md) gives the official download links, terms and the expected layout under
`$CORPUS_ROOT`; no imagery is redistributed. [docs/FILTERING.md](docs/FILTERING.md) describes the pair
eligibility rules, the six quality-filter categories (featureless content, cloud contamination,
misregistration, compression artifacts, missing tiles, dead/no-data regions) and the crop-equivalent source
sampling. The resulting decisions are published as per-stream keep tables in
[`corpus/keep_v1/`](corpus/keep_v1) with counts and mixture weights in
[`corpus/MANIFEST.json`](corpus/MANIFEST.json): 3,281,393 eligible pairs, of which 3,204,744 are kept for
Stage 2. Stage 2 reads these tables directly.

## 🏋️ Training

```bash
# released weights (the frozen autoencoder of every stage and the Stage 3 starting point)
hf download JeonghyeokDo/GeoSET --include "generalist/*" --local-dir weights/GeoSET
# FLUX.2-klein-base-4B, used only to initialise Stage 2
hf download black-forest-labs/FLUX.2-klein-base-4B flux-2-klein-base-4b.safetensors \
    --revision a3b4f4849157f664bdbc776fd7453c2783562f4d --local-dir weights/flux2-klein-base-4b

# Stage 1: speckle-robust SAR encoder (4 GPUs)
accelerate launch --num_processes 4 --mixed_precision no --dynamo_backend no scripts/train_stage1.py \
    --corpus-root $CORPUS_ROOT --vae weights/GeoSET/generalist/vae --out runs/stage1

# Stage 2: generalist pretraining (4 GPUs)
accelerate launch --num_processes 4 --mixed_precision no --dynamo_backend no scripts/train_stage2.py \
    --flux2-base weights/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors --corpus-root $CORPUS_ROOT \
    --sar-encoder runs/stage1/checkpoints/step_0050000 --vae weights/GeoSET/generalist/vae --out runs/stage2

# Stage 3: LoRA adaptation (1 GPU)
accelerate launch --num_processes 1 --mixed_precision no --dynamo_backend no scripts/train_stage2.py \
    --downstream sar2opt --data-root $DATA_ROOT --lora --init-from weights/GeoSET/generalist/transformer \
    --sar-encoder weights/GeoSET/generalist/sar_encoder --vae weights/GeoSET/generalist/vae --out runs/lora/sar2opt

# Stage 3: full fine-tuning (1 GPU)
accelerate launch --num_processes 1 --mixed_precision no --dynamo_backend no scripts/train_stage2.py \
    --downstream sar2opt --data-root $DATA_ROOT --full-finetune --init-from weights/GeoSET/generalist/transformer \
    --sar-encoder weights/GeoSET/generalist/sar_encoder --vae weights/GeoSET/generalist/vae --out runs/full/sar2opt
```

`--downstream` takes `qxs-saropt`, `sar2opt`, `sar2eo` or `spacenet6`. Stage 2 can also start from the released
SAR encoder (`--sar-encoder weights/GeoSET/generalist/sar_encoder`). The argument defaults are the paper's
settings:

| | Stage 1 | Stage 2 | Stage 3, LoRA | Stage 3, full FT |
|---|---|---|---|---|
| Trained | SAR encoder | all 3.852B generator parameters | 23.10M adapter parameters (0.60%) | all 3.852B generator parameters |
| Data | SAR images of the corpus | 3,204,744 SAR–EO pairs | downstream training split | downstream training split |
| Updates | 50,000 | 500,000 | 20,000 | 20,000 |
| Global batch size | 432 | 256 | 16 | 16 |
| Learning rate (AdamW) | 10⁻⁵, 500-update warmup | 10⁻⁴ peak, 1,000-update warmup | 10⁻⁴, rank 16, α 16 | 2 × 10⁻⁵ |
| EMA decay | – | 0.9999 | 0.9999 | 0.9999 |
| Hardware | 4 × NVIDIA B200 | 4 × NVIDIA B200 | 1 GPU | 1 GPU |
| Training time | 14.8 h | 129.3 h | 0.98 h per dataset | 1.34 h per dataset |

All stages train on 256 × 256 crops and clip the gradient norm to 1.0; Stages 2 and 3 drop the SAR condition
with probability 0.1 for classifier-free guidance. Each run writes `<out>/checkpoints/step_XXXXXXX/` (the EMA
transformer, or the EMA adapter with `--lora`) and a resumable state (`--resume`). Use a Stage 3 result with
`translate.py --lora runs/lora/sar2opt/checkpoints/step_0020000` or
`translate.py --transformer runs/full/sar2opt/checkpoints/step_0020000`.

## 📏 Evaluation

`scripts/evaluate.py` scores a folder of translated images (`<stem>.png`, one per test item) on a frozen test
split with the paper's seven metrics: **FID** and **DISTS** (primary), KID, DINO similarity (frozen
DINOv3-SAT ViT-L/16), LPIPS, SSIM and PSNR. It is standalone, so the outputs of the comparison methods are
scored by the same code.

```bash
# DINO similarity needs the DINOv3 code and the SAT-493M ViT-L/16 weights (Meta's DINOv3 release, DINOv3 License)
git clone https://github.com/facebookresearch/dinov3 && git -C dinov3 checkout 31703e4cbf1ccb7c4a72daa1350405f86754b6d1

python scripts/evaluate.py --dataset sar2opt --pred-dir outputs/geoset-lora/sar2opt --data-root $DATA_ROOT \
    --dino-repo dinov3 --dino-weights dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

Predictions must have the evaluation size listed in Data Preparation (SAR2Opt: 512 × 512); the reference is
center-cropped to it. Keep the weights file name as released. Without `--dino-repo/--dino-weights`, DINO is
skipped. The row is printed in the paper's column order and saved to `<pred-dir>.json` (`--out` to change it).
The paper's scores were computed on a CUDA GPU with default PyTorch settings; on a CPU, FID and KID can
differ in the last printed digit.

## 🧪 Comparison Methods

[`baselines/`](baselines/README.md) reproduces the sixteen comparison methods of Tables 2 and 3 of the paper,
retrained on the same training splits and evaluated on the same test pairs with `scripts/evaluate.py`: 61
checkpoints over the four benchmarks (Seg-CycleGAN on SpaceNet6 only). Each method folder fetches the upstream
code at a pinned commit, applies our patch where the code was changed, and provides the configurations and
training and test scripts. [MODEL_ZOO.md](MODEL_ZOO.md) lists the metrics, Hub folder and licence of every
checkpoint; the weights are under
[`baselines/<dataset>/<method>/`](https://huggingface.co/JeonghyeokDo/GeoSET/tree/main/baselines) in the
Hugging Face repository.

```bash
export BASELINES_ROOT=/path/to/upstream_code DATA_ROOT=/path/to/data WORK_DIR=/path/to/work CKPT_ROOT=/path/to/checkpoints
bash baselines/setup.sh                        # clone every method at its pinned commit and apply its patch
bash baselines/<method>/train.sh <dataset>     # after building the method's data view (baselines/README.md)
bash baselines/<method>/test.sh <dataset>
```

## 📁 Repository Structure

```
GeoSET/
├── src/geoset/
│   ├── transformer_geoset.py      # GeoSETTransformer2DModel: SAR-conditioned flow transformer
│   ├── sar_encoder_geoset.py      # GeoSETSAREncoder: speckle-robust SAR encoder
│   ├── autoencoder_flux2.py       # AutoencoderFlux2: frozen FLUX.2 autoencoder
│   ├── pipeline_geoset.py         # GeoSETPipeline (diffusers)
│   ├── scheduler_flow_bridge.py   # FlowBridgeScheduler: Euler integration from t = 0 to t = 1
│   ├── lora.py                    # LoRA adapters
│   ├── speckle.py                 # domain-aware speckle augmentation
│   └── data/                      # pretraining corpus and downstream loaders
├── scripts/
│   ├── train_stage1.py            # Stage 1: SAR encoder
│   ├── train_stage2.py            # Stage 2: generalist pretraining; Stage 3: --lora / --full-finetune
│   ├── translate.py               # inference on a folder or a benchmark test split
│   ├── evaluate.py                # the seven metrics
│   ├── datasets/                  # SpaceNet6 chip builder
│   └── corpus/                    # pair lists, EO scoring and keep-table builder
├── corpus/                        # keep tables and MANIFEST.json of the pretraining corpus
├── splits/                        # frozen train/test lists of the four benchmarks
├── docs/                          # CORPUS.md, FILTERING.md, DATASETS.md
├── baselines/                     # the sixteen comparison methods
├── assets/                        # figures of this README
├── MODEL_ZOO.md                   # weights and metrics of every released checkpoint
├── LICENSE                        # Apache-2.0, the code
├── LICENSE-WEIGHTS.md             # licences of the weights
├── NOTICE                         # third-party attributions
└── requirements.txt
```

## 🚀 Code Release Plan

- ✅ Inference code
- ✅ Pretrained models
- ✅ Training scripts
- ✅ Evaluation scripts

## 📑 Citation
If you find GeoSET useful, please consider citing:
```BibTeX
@article{do2026geoset,
  title={GeoSET: Generalist Foundation Model for SAR-to-EO Image Translation},
  author={Do, Jeonghyeok and Kim, Munchurl},
  journal={arXiv preprint arXiv:2609.37496},
  year={2026}
}
```

Our prior work on SAR-to-EO image translation, [C-DiffSET](https://github.com/KAIST-VICLab/C-DiffSET) ([project page](https://kaist-viclab.github.io/C-DiffSET_site/)):
```BibTeX
@article{do2026cdiffset,
  title={C-diffset: Leveraging latent diffusion for sar-to-eo image translation with confidence-guided reliable object generation},
  author={Do, Jeonghyeok and Lee, Jaehyup and Lee, Seungchul and Kim, Munchurl},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

## 📜 License

The code is released under the [Apache License 2.0](LICENSE), with third-party attributions in
[NOTICE](NOTICE). The GeoSET weights (generalist transformer, SAR encoder and LoRA adapters) are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); the bundled FLUX.2-klein-base-4B
autoencoder remains under Apache-2.0; each comparison method's weights carry the terms of the code it was
trained with ([MODEL_ZOO.md](MODEL_ZOO.md)). See [LICENSE-WEIGHTS.md](LICENSE-WEIGHTS.md). The datasets are not
redistributed; their terms remain with their owners.

The transformer and autoencoder code is adapted from [FLUX.2](https://github.com/black-forest-labs/flux2) by
Black Forest Labs (Apache-2.0), and the generator is initialised from
[FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B). GeoSET is not a FLUX
product and is not endorsed by Black Forest Labs.
