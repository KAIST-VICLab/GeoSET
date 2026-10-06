# Frozen splits

These are the exact train/test lists used for every downstream result in the paper. Each line is the
path of one **EO** image relative to the dataset directory `$DATA_ROOT/<dataset>/`; the SAR path is
derived from it by a fixed rule, so the two sides cannot drift apart. No imagery is included.

| Dataset | Lines look like | SAR path rule |
|---|---|---|
| `qxs-saropt` | `opt_256_oc_0.2/<n>.png` | `opt_256_oc_0.2/` → `sar_256_oc_0.2/` |
| `sar2opt` | `trainB/<name>.jpg`, `testB/<name>.jpg` | last letter of the folder `B` → `A` (`testB` → `testA`) |
| `sar2eo` | `SAR-EO/EO/train/Gotcha<k>.png`, `SAR-EO/EO/validation/Gotcha<k>.png` | `/EO/` → `/SAR/` |
| `spacenet6` | `train/AOI_11_Rotterdam/PS-RGB/SN6_Train_AOI_11_Rotterdam_PS-RGB_<...>.tif` | `PS-RGB` → `SAR-Intensity` |

| List | Items | sha256 |
|---|---:|---|
| `qxs-saropt/train.txt` | 16,001 | `80489fba906b7e2cb1f21ad05b3306351750b11982824ffa7b3ec952c28dce8d` |
| `qxs-saropt/test.txt` | 3,999 | `bee3faadf537dfab8ec1dc94b6655ea4628c50d80335fe6e87c2787d7fc5d474` |
| `sar2opt/train.txt` | 1,450 | `c37f4a888adc325712b793dea3338b6094e270b161ec77d2cefc03d4255fa7db` |
| `sar2opt/test.txt` | 627 | `237a9d5f357160a948db4b7d27fe46b65d3417dd521082a9a23c5f0af8df4366` |
| `sar2eo/train.txt` | 68,151 | `7f8d01021174caa292c3996c822db34089338404d0eccf964162bf6340df84c1` |
| `sar2eo/test.txt` | 21,260 | `7bd1583df52f478b90f1f88649ff0d61631f9aba48e9cc769075f87f2fcc5434` |
| `sar2eo/test_first4000.txt` | 4,000 | `fc9c4747377b2e5081fdf8edf0d59839442b9e052a3deb88da2194deb1d3bdd8` |
| `spacenet6/train.txt` | 2,558 | `8720d40d11fd6dc2288c1cd0467a31252e3f3ed43e81dbded6a39bf92672b797` |
| `spacenet6/test.txt` | 495 | `e154dacbf982cfa5946225cad26ddcf0cda0e7b447a6f14ad78919beaa48054f` |
| `spacenet6/sar_stretch.json` | – | `00bd1ab1e3fe1ab59f93660e20da12c97a9df0ec4d5a9132266ee6bf6e692ff4` |

## SpaceNet6 chips

The `spacenet6` lists name raw GeoTIFFs of the SpaceNet6 training archive. They are rendered once into
256-px chips with the fixed per-band SAR window in `spacenet6/sar_stretch.json`:

```bash
python scripts/datasets/build_spacenet6.py --data-root $DATA_ROOT --verify-stretch
```

This writes `$DATA_ROOT/spacenet6/chips/{trainA,trainB,testA,testB}/<stem>.png` (`A` = SAR, `B` = EO,
`<stem>` = the EO file name without `_PS-RGB` and `.tif`); training and evaluation read the chips. A rebuild
can be checked against the pixel digests below (sha256 over the chips of one folder in file-name order, each
contributing its file name, `str(shape)` of the decoded uint8 array and the raw pixel bytes):

| Folder | Chips | Pixel digest |
|---|---:|---|
| `chips/trainA` | 2,558 | `0070c62bd124960a8ebbbfc39bdff10eea43a8be9174b5ef7a7b07da62ecbdf7` |
| `chips/trainB` | 2,558 | `04a25df47eb30a1d118ab3a75ac31d3973a96a35ae3d615be36c9c77aef76733` |
| `chips/testA` | 495 | `aa68f5bc549cf2185b3e9277c6551226ef2b72c0fea4243bbff03bf8bde6d602` |
| `chips/testB` | 495 | `92cf94b769e1f154621ab069e8cd994c958a3551a4153e6851cd10b66456dcdc` |

## Protocols (paper, Table 10)

- **QXS-SAROPT** uses the C-DiffSET splits.
- **SAR2Opt** uses the official train/test folders; its 600-px test tiles are evaluated on the
  deterministic center 512 × 512 crop.
- **SAR2EO** uses the E3Diff split (`SAR-EO/*/train` and `SAR-EO/*/validation`). Reported numbers use the
  fixed first 4,000 samples of the 21,260-pair test split, `sar2eo/test_first4000.txt` (the first 4,000
  lines of `sar2eo/test.txt`).
- **SpaceNet6** uses UTM-easting blocks with a 450 m guard band between train and test.
