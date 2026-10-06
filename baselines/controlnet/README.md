# ControlNet

Zhang, Rao and Agrawala, "Adding conditional control to text-to-image diffusion models", ICCV 2023.
A ControlNet on Stable Diffusion 2.1-base takes the SAR image as its conditioning image and generates the EO
image from the fixed prompt `"electro-optical image"`.

| | |
|---|---|
| Training script | `train_controlnet.py`: the diffusers example [`examples/controlnet/train_controlnet.py`](https://github.com/huggingface/diffusers/blob/v0.37.0/examples/controlnet/train_controlnet.py) (v0.37.0), Apache-2.0, with one modification (below) |
| Inference script | `infer_controlnet.py` (ours) |
| Base model | `Manojb/stable-diffusion-2-1-base` (Stable Diffusion 2.1-base), revision `0094d483a120f3f33dafbd187ea4aa60d10de75c` |
| Requirements | diffusers 0.37, transformers, accelerate, datasets |

The modification of `train_controlnet.py`: a `conditioning_image` column that holds file paths is cast to
`datasets.Image()` so that the training transforms receive images. Nothing needs to be fetched; both scripts run
from this folder.

## Data

```bash
python baselines/data/ldm_prepare.py <dataset>    # env: DATA_ROOT WORK_DIR
```

This writes the image folder `$WORK_DIR/ldm_data/<dataset>/controlnet/train/`: links to the EO training images and
`metadata.jsonl` with the prompt and the absolute path of the paired SAR image (SAR2Opt: centre 512 × 512 PNG
crops; SpaceNet6: the chips of `scripts/datasets/build_spacenet6.py`). Test inputs: `$WORK_DIR/ldm_data/<dataset>/test_sar`
(`sar2eo`: the first 4,000 test pairs).

## Training and testing

Environment: `WORK_DIR`, `CKPT_ROOT`, optional `GPU` (default 0) and `PY` (default `python`).

```bash
bash baselines/controlnet/train.sh <dataset>   # run in $WORK_DIR/controlnet/<dataset>, weights -> $CKPT_ROOT/controlnet/<dataset>/
bash baselines/controlnet/test.sh <dataset>    # -> $WORK_DIR/results/controlnet/<dataset>/<stem>.png
python scripts/evaluate.py --dataset <dataset> --pred-dir $WORK_DIR/results/controlnet/<dataset>
```

| Dataset | Resolution | Batch (training and testing) |
|---|---:|---:|
| `qxs-saropt`, `sar2eo`, `spacenet6` | 256 | 32 |
| `sar2opt` | 512 | 8 |

Training: 50,000 updates, AdamW, constant learning rate 1e-5, bf16 mixed precision,
seed 42; the Stable Diffusion UNet, VAE and text encoder stay frozen and the ControlNet is initialised from the
UNet. Testing: UniPC, 50 steps, guidance scale 7.5, bf16. One generator seeded with 42 is consumed batch by
batch, so the outputs depend on the batch size in the table.

## Released checkpoints

`JeonghyeokDo/GeoSET`: `baselines/<dataset>/controlnet/{config.json,diffusion_pytorch_model.safetensors}`. Copy both
files to `$CKPT_ROOT/controlnet/<dataset>/` and run `test.sh`.
