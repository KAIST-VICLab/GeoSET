# Downstream datasets

GeoSET is adapted and evaluated on four public SAR–EO benchmarks. No imagery is redistributed here: download
each dataset from its official source under its own terms and arrange it under one directory `$DATA_ROOT`.
The frozen train/test lists used for every result in the paper are in [`splits/`](../splits/README.md).
CAP-BSG and KOMPSAT, the two private datasets reported in the paper's appendix, are not distributed.

| Dataset | Key | SAR source | EO source | GSD | Train / test | Eval. size | Split protocol |
|---|---|---|---|---|---:|---|---|
| QXS-SAROPT (Huang et al., 2021) | `qxs-saropt` | GF-3 | Google Earth | 1 m | 16,001 / 3,999 | 256² | C-DiffSET splits (Do et al., 2026) |
| SAR2Opt (Zhao et al., 2022) | `sar2opt` | TerraSAR-X | Google Earth | 1 m | 1,450 / 627 | 600² → 512² | official train/test folders |
| SAR2EO (Du et al., 2023) | `sar2eo` | airborne UNICORN SAR | monochrome WAMI | – | 68,151 / 4,000 | 256² | E3Diff split (Qin et al., 2024) |
| SpaceNet6 (Shermeyer et al., 2020) | `spacenet6` | airborne Capella X-band | WorldView-2 | 0.5 m | 2,558 / 495 | 256² | UTM-easting blocks; 450 m guard band |

SAR2Opt is evaluated on the deterministic center 512 × 512 crop of its 600-px test tiles. SAR2EO uses the
fixed first 4,000 samples of the 21,260-pair E3Diff test split.

## Directory layout

```
$DATA_ROOT/
  qxs-saropt/{opt_256_oc_0.2,sar_256_oc_0.2}/<n>.png
  sar2opt/{trainA,trainB,testA,testB}/<name>.jpg                   # A = SAR, B = EO
  sar2eo/SAR-EO/{EO,SAR}/{train,validation}/Gotcha<k>.png
  spacenet6/train/AOI_11_Rotterdam/{PS-RGB,SAR-Intensity,geojson_buildings}/...
  spacenet6/chips/{trainA,trainB,testA,testB}/<stem>.png         # written by scripts/datasets/build_spacenet6.py
```

Every script of this repository, including the comparison methods in [`baselines/`](../baselines/README.md),
reads this layout through `--data-root $DATA_ROOT` or the `DATA_ROOT` environment variable.

## QXS-SAROPT

- **Source**: <https://github.com/yaoxu008/QXS-SAROPT>. The download link is shown after completing the
  questionnaire linked from that page (<https://www.wenjuan.com/s/UZBZJv5GwL/>).
- **Content**: 20,000 GaoFen-3 SAR / Google Earth optical pairs, 256 × 256 PNG.
- **Terms**: the repository states no licence; its README asks that the dataset paper be cited when the
  dataset is used for research.
- **Prepare**: place the two folders of the download, `opt_256_oc_0.2` (EO) and `sar_256_oc_0.2` (SAR), in
  `$DATA_ROOT/qxs-saropt/`.
- **Split**: `splits/qxs-saropt/{train,test}.txt`, the C-DiffSET splits (16,001 / 3,999).

## SAR2Opt

- **Source**: <https://github.com/MarsZhaoYT/SAR2Opt-Heterogeneous-Dataset>, download links on that page:
  [Google Drive](https://drive.google.com/file/d/1XB9pWq-tVdxQsbVALxbYIF0Em90J4kkR/view?usp=sharing) or
  [Baidu Netdisk](https://pan.baidu.com/s/1xQ1nc2aPFdJ99SI2upl5Tg) (code `hy8d`).
- **Content**: TerraSAR-X SAR / Google Earth optical pairs, 600 × 600 JPEG.
- **Terms**: the repository is released under the MIT licence; the dataset archive is distributed through the
  links in its README.
- **Prepare**: arrange the four folders as `$DATA_ROOT/sar2opt/{trainA,trainB,testA,testB}` (`A` = SAR,
  `B` = EO, same file names on both sides).
- **Split**: `splits/sar2opt/{train,test}.txt`, the official folders (1,450 / 627).

## SAR2EO

- **Source**: the SAR-EO paired set released by the E3Diff authors, linked from
  <https://github.com/DeepSARRS/E3Diff> ("SAR-EO dataset"):
  [Baidu Netdisk](https://pan.baidu.com/s/1N6ynB-au7FO8of7rljBIGw?pwd=gn72) (code `gn72`). The release is a
  seven-part RAR5 archive, `SAR-EO.part1.rar` … `SAR-EO.part7.rar` (7,186,526,021 bytes in total).
- **Content**: airborne UNICORN SAR / monochrome WAMI EO pairs, 256 × 256 single-channel PNG.
- **Terms**: no licence is stated for this release.
- **Prepare**: extract with 7-Zip (`bsdtar` does not read this multi-volume archive):

  ```bash
  mkdir -p $DATA_ROOT/sar2eo && 7zz x SAR-EO.part1.rar -o$DATA_ROOT/sar2eo
  ```

  This gives `SAR-EO/{EO,SAR}/train` (68,151 pairs) and `SAR-EO/{EO,SAR}/validation` (21,260 pairs). The
  archive also contains macOS metadata (`__MACOSX/`, `.DS_Store`), which is not read.
- **Split**: `splits/sar2eo/{train,test}.txt`, the E3Diff split (`train` / `validation` folders). Reported
  numbers use `splits/sar2eo/test_first4000.txt`, the first 4,000 lines of `test.txt`.

## SpaceNet6

- **Source**: SpaceNet 6: Multi-Sensor All-Weather Mapping (<https://spacenet.ai/sn6-challenge/>), on the
  Registry of Open Data on AWS (no account needed). Only the training archive is used:

  ```bash
  aws s3 cp --no-sign-request \
      s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz .
  ```

- **Content**: aerial X-band SAR (`SAR-Intensity`, four polarizations) and pan-sharpened WorldView-2 RGB
  (`PS-RGB`), 900 × 900 tiles at 0.5 m, plus building footprints (`geojson_buildings`).
- **Licence**: The SpaceNet Dataset by SpaceNet Partners is licensed under a Creative Commons
  Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
- **Prepare**: extract the three folders that are read (`geojson_buildings` is used only by the
  Seg-CycleGAN baseline):

  ```bash
  mkdir -p $DATA_ROOT/spacenet6
  tar -xzf SN6_buildings_AOI_11_Rotterdam_train.tar.gz -C $DATA_ROOT/spacenet6 \
      ./train/AOI_11_Rotterdam/PS-RGB ./train/AOI_11_Rotterdam/SAR-Intensity ./train/AOI_11_Rotterdam/geojson_buildings
  ```

- **Split**: `splits/spacenet6/{train,test}.txt`: UTM-easting blocks with a 450 m guard band between train
  and test (2,558 / 495 tiles of the training archive).
- **Chips**: training and evaluation read 256-px chips rendered once from the GeoTIFFs (needs `rasterio`):

  ```bash
  python scripts/datasets/build_spacenet6.py --data-root $DATA_ROOT --verify-stretch
  ```

  EO tiles are resized to 256 × 256 (bicubic). The HH, HV and VV bands of `SAR-Intensity` are mapped from
  `10 * log10(x)` to 0–255 with the fixed per-band window in `splits/spacenet6/sar_stretch.json`, then
  resized to 256 × 256 (bilinear). `--verify-stretch` re-derives the window from the training tiles and
  checks it against the file. The script writes `$DATA_ROOT/spacenet6/chips/{trainA,trainB,testA,testB}/<stem>.png`
  (`A` = SAR, `B` = EO, `<stem>` = the EO file name without `_PS-RGB` and `.tif`).

## Checksums

File-set digests are the SHA-256 of the `sha256sum` listing of the files in byte-order path order. Run the
commands from the repository root:

```bash
S=$PWD/splits
# QXS-SAROPT, all 40,000 files
(cd $DATA_ROOT/qxs-saropt && find opt_256_oc_0.2 sar_256_oc_0.2 -type f | LC_ALL=C sort | xargs -d '\n' sha256sum | sha256sum)
# QXS-SAROPT, the 3,999 test pairs
(cd $DATA_ROOT/qxs-saropt && { cat $S/qxs-saropt/test.txt; sed 's|^opt_256_oc_0.2/|sar_256_oc_0.2/|' $S/qxs-saropt/test.txt; } \
    | LC_ALL=C sort | xargs -d '\n' sha256sum | sha256sum)
# SAR2Opt, all 4,154 files
(cd $DATA_ROOT/sar2opt && find trainA trainB testA testB -type f | LC_ALL=C sort | xargs -d '\n' sha256sum | sha256sum)
# SAR2EO, all 178,822 PNG files
(cd $DATA_ROOT/sar2eo/SAR-EO && find EO/train EO/validation SAR/train SAR/validation -maxdepth 1 -type f -name '*.png' \
    | LC_ALL=C sort | xargs -d '\n' sha256sum | sha256sum)
# SAR2EO, the first 4,000 test pairs
(cd $DATA_ROOT/sar2eo/SAR-EO && { sed 's|^SAR-EO/||' $S/sar2eo/test_first4000.txt; sed 's|^SAR-EO/EO/|SAR/|' $S/sar2eo/test_first4000.txt; } \
    | LC_ALL=C sort | xargs -d '\n' sha256sum | sha256sum)
```

| Files | Count | SHA-256 |
|---|---:|---|
| QXS-SAROPT, all | 40,000 | `3786bc7c986f47a14a2d7a19bbb971846cd61bf198cf3f7ccb95366e8f256d46` |
| QXS-SAROPT, test pairs | 7,998 | `4e1a3767292cb052a8d3369a3f3648bc0cfa103da57e2bd6cca45b608c6a3df4` |
| SAR2Opt, all | 4,154 | `563305e1462d8ab7a1fb36c91205338ae870d3c591965e26fd647130d20ca271` |
| SAR2EO, all | 178,822 | `30a4c9c9f3ccc653ab88a38d483860969bb8ca3113a506ba8615e0e3034393f2` |
| SAR2EO, first 4,000 test pairs | 8,000 | `fc4f298ee0e65d2b36c676df3494db81b27f723f54f91dd9cbb18f9d33f6cca3` |
| `SN6_buildings_AOI_11_Rotterdam_train.tar.gz` (41,874,641,730 bytes) | 1 | `406e5b7763efc13f5a304b53fafd32eacab0b53e516b6a2804922fa9eef7a9e6` |

The SpaceNet6 chips can be checked against the pixel digests in [`splits/README.md`](../splits/README.md#spacenet6-chips),
computed per chip folder with:

```python
import hashlib, sys
from pathlib import Path
import numpy as np
from PIL import Image

h = hashlib.sha256()
for p in sorted(Path(sys.argv[1]).glob("*.png")):          # e.g. $DATA_ROOT/spacenet6/chips/testA
    a = np.asarray(Image.open(p))
    h.update(p.name.encode() + str(a.shape).encode() + a.tobytes())
print(h.hexdigest())
```

## References

- Meiyu Huang, Yao Xu, Lixin Qian, Weili Shi, Yaqin Zhang, Wei Bao, Nan Wang, Xuejiao Liu, and Xueshuang
  Xiang. The QXS-SAROPT dataset for deep learning in SAR-optical data fusion. arXiv preprint
  arXiv:2103.08259, 2021.
- Yitao Zhao, Turgay Celik, Nanqing Liu, and Heng-Chao Li. A comparative analysis of GAN-based methods for
  SAR-to-optical image translation. IEEE Geoscience and Remote Sensing Letters, 19:1–5, 2022.
- Shenshen Du, Jun Yu, Guochen Xie, Renjie Lu, Pengwei Li, Zhongpeng Cai, and Keda Lu. SAR2EO: A
  high-resolution image translation framework with denoising enhancement. In Australasian Joint Conference
  on Artificial Intelligence, pp. 91–102. Springer, 2023.
- Jiang Qin, Bin Zou, Haolin Li, and Lamei Zhang. Efficient end-to-end diffusion model for one-step
  SAR-to-optical translation. IEEE Geoscience and Remote Sensing Letters, 22:1–5, 2024.
- Jacob Shermeyer, Daniel Hogan, Jason Brown, Adam Van Etten, Nicholas Weir, Fabio Pacifici, Ronny Hänsch,
  Alexei Bastidas, Scott Soenen, Todd Bacastow, et al. SpaceNet 6: Multi-sensor all weather mapping dataset.
  In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 768–777, 2020.
- Jeonghyeok Do, Jaehyup Lee, Seungchul Lee, and Munchurl Kim. C-DiffSET: Leveraging latent diffusion for
  SAR-to-EO image translation with confidence-guided reliable object generation. IEEE Transactions on
  Circuits and Systems for Video Technology, 2026.
