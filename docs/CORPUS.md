# Pretraining corpus

GeoSET is pretrained on five SAR–EO dataset families that form six streams, with 3MOS divided into
`3mos_mr` (mid-resolution) and `3mos_hr` (high-resolution). No imagery is redistributed here: download each
dataset from its official source under its own terms, arrange it under one directory `$CORPUS_ROOT`, and
use the keep tables in [`corpus/keep_v1/`](../corpus/keep_v1) to select the stage-2 pairs
([FILTERING.md](FILTERING.md)).

Stage 1 uses only SAR images and requires neither paired EO images nor cross-modal alignment; it reads the
SAR side of every stream. Stage 2 reads the kept SAR–EO pairs.

| stream | dataset | SAR source | EO source | spatial scale | tile size | SAR input |
|---|---|---|---|---|---|---|
| `guso` | GUSO (Yan et al., 2026) | ICEYE, Capella, Umbra | supplied optical imagery | 0.16–0.98 m | 512² | 1-ch display |
| `terramesh` | TerraMesh (Blumenstiel et al., 2025) | Sentinel-1 RTC, VV/VH | Sentinel-2 L2A RGB | 10 m grid | 264² | 2-ch dB |
| `sarlo80` | SARLO-80 (Debuysère et al., 2026) | Umbra spotlight SLC | Google satellite imagery | 0.8 m slant-range grid | 1024² | 1-ch amplitude |
| `sar1m` | SAR-1M (Liu et al., 2026) | multi-source paired subset | paired optical counterparts | source-dependent | 256² | 1-ch display |
| `3mos_mr` | 3MOS (Ye et al., 2025) | Sentinel-1, ALOS-family | Google Earth | 10 / 12.5 m | 256² | 1-ch display |
| `3mos_hr` | 3MOS (Ye et al., 2025) | GF-3, RADARSAT-2, RCM | Google Earth | 3.5 / 6.3 / 12.5 m | 256² | 1-ch display |

## Directory layout

```
$CORPUS_ROOT/
  guso/<region>/<split>/<Continent>/<site>_<Continent>_<label>_<idx6>_{SAR,OPT}.tif
  terramesh/{train,val}_metadata.parquet
  terramesh/{train,val}/{S1RTC,S2RGB}/majortom_shard_NNNNNN.tar     # members majortom_<split>_NNNNNNN.zarr.zip
  sarlo80/train_x/chunk_XXX/shard-YYYYY/<key>.{sar.npy,meta.json,sicd.xml,...}
  sarlo80/optic/chunk_XXX/shard-YYYYY/<key>.optic.png                # reconstructed optical images (stage 2)
  sar1m/{SAR,OPT}/<name>
  sar1m/paired.json
  3mos/{ALOS,GF3,RCM,Radarsat}/{sar/sar_<id>.jpg, opt/<terrain>/opt_<id>.jpg}
  3mos/SEN/SEN1_region{1..7}/{sar/sar_<id>.jpg, opt/<terrain>/opt_<id>.jpg}
```

The data loaders (`geoset.data.SARCorpus` for stage 1, `geoset.data.PairedCorpus` for stage 2) read this
layout from `$CORPUS_ROOT`. The commands below use the `hf` command-line tool of `huggingface_hub`.

## GUSO

- **Source**: <https://github.com/vision-heng/GUSO>. Fill out the request form linked there; download links
  (Google Drive and Baidu Netdisk) for the five general-scene datasets are shown after submission.
- **Terms**: academic use only, commercial use prohibited; any secondary development or modification based
  on GUSO requires prior written consent from RSIDEA, Wuhan University (see the GUSO README).
- **Download**: the five general-scene categories for all three splits, 15 archives
  `{hill,plain,rural,urban,water}-{train,val,test}.zip`.
- **Prepare**: extract each archive into its category directory:

  ```bash
  for z in {hill,plain,rural,urban,water}-{train,val,test}.zip; do
    unzip -q "$z" -d "$CORPUS_ROOT/guso/${z%%-*}"
  done
  ```

  This gives 589,143 `*_SAR.tif` tiles, of which 589,137 have an `*_OPT.tif` counterpart (the remaining six
  lie under `unpaired_temp/`).

## TerraMesh

- **Source**: <https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh> (not gated); the keep table was
  verified against revision `9f13e7172f16e7e04bae8c303fe10e32dcde845d`.
- **Licence**: CC-BY-SA-4.0.
- **Download**: the Major TOM shards of the `S1RTC` and `S2RGB` modalities for `train` and `val`, and the two
  metadata files (about 3.05 TB):

  ```bash
  REV=9f13e7172f16e7e04bae8c303fe10e32dcde845d
  for p in "train/S1RTC/majortom_*" "train/S2RGB/majortom_*" "val/S1RTC/majortom_*" "val/S2RGB/majortom_*" \
           "*_metadata.parquet"; do
    hf download ibm-esa-geospatial/TerraMesh --repo-type dataset --revision $REV --include "$p" \
        --local-dir "$CORPUS_ROOT/terramesh"
  done
  ```

  The tars are read in place. At this revision the Major TOM subset holds 7,971,665 train and 80,384 val
  samples. The keep table addresses samples by `<split>/<zarr member name>`; 1,709,540 of the 1,741,434
  kept TerraMesh pairs are distributed at this revision, and members that are not present are skipped.

## SARLO-80

- **Source**: <https://huggingface.co/datasets/ONERA/SARLO-80> (not gated), revision
  `96f13557b6610ea70e7023ce8b869d8a50804400`: 519 WebDataset shards, 87,870 samples (828 GB).
- **Licence**: CC-BY-SA-4.0.
- **Download and extract**:

  ```bash
  hf download ONERA/SARLO-80 --repo-type dataset --revision 96f13557b6610ea70e7023ce8b869d8a50804400 \
      --include "train/*" --local-dir "$CORPUS_ROOT/sarlo80"
  cd "$CORPUS_ROOT/sarlo80"
  for t in train/chunk_*/shard-*.tar; do
    d="train_x/${t#train/}"; d="${d%.tar}"
    mkdir -p "$d" && tar -xf "$t" -C "$d"
  done
  ```

- **Optical images (stage 2 only)**: SARLO-80 does not redistribute optical images. Reconstruct them by
  following SARLO-80's official instructions,
  [README_OPTICAL_RECONSTRUCTION.md](https://huggingface.co/datasets/ONERA/SARLO-80/blob/96f13557b6610ea70e7023ce8b869d8a50804400/README_OPTICAL_RECONSTRUCTION.md),
  and save, for every sample, the optical image projected into the SAR slant-range crop frame (same size as
  the SAR crop) as `$CORPUS_ROOT/sarlo80/optic/<chunk>/<shard>/<key>.optic.png`. As SARLO-80 states, users
  are responsible for ensuring that their access to and use of the optical tile source named in the metadata
  complies with that provider's terms of use.

## SAR-1M

- **Source**: <https://huggingface.co/datasets/Wenquandan777/SAR-1M> (gated: log in and accept the
  conditions), revision `0c1b5595df080bd9bb27f6368cf8dd642f44c7bc`, file `SAR-1M_DATA.zip` (76.6 GB,
  SHA-256 `3aa768c5d3ba3b3ce9bbd1003634e5eea057aca257aa2784b900e8760a7cc358`). Project page:
  <https://github.com/MiliLab/SARMAE>.
- **Licence**: CC BY-NC 4.0, non-commercial research only.
- **Download and extract**:

  ```bash
  hf auth login
  hf download Wenquandan777/SAR-1M SAR-1M_DATA.zip --repo-type dataset \
      --revision 0c1b5595df080bd9bb27f6368cf8dd642f44c7bc --local-dir "$CORPUS_ROOT/sar1m"
  ```

  Extract the archive so that `$CORPUS_ROOT/sar1m` contains `SAR/`, `OPT/` and `paired.json`. Only the
  731,080 entries of `paired.json` are read, in both stages.

## 3MOS

- **Source**: <https://github.com/3M-OS/3MOS> (archive `3MOS.rar`, 6.87 GB, via the Baidu NetDisk or
  OneDrive link on that page). The archive used for the keep tables has SHA-256
  `530ffef637a695188638382270605fcc8db9ff8bfd071c11e38f74d39331de59`.
- **Licence**: CC BY-NC-ND 4.0; the 3MOS README lists the terms of the underlying SAR and optical imagery.
- **Prepare**: `unrar x 3MOS.rar "$CORPUS_ROOT/3mos/"`, which gives the sensor-group folders `ALOS`, `GF3`,
  `RCM`, `Radarsat` and `SEN/SEN1_region1` to `SEN1_region7` with 113,074 SAR and 113,074 optical images.
  `3mos_mr` is `SEN/SEN1_region1-7` and `ALOS` (87,293 pairs); `3mos_hr` is `GF3`, `Radarsat` and `RCM`
  (25,781 pairs).

## References

- Heng Yan, Ailong Ma, Hong Shu, Yuting Wan, Liangpei Zhang, and Yanfei Zhong. Ultra-high-resolution SAR and
  optical image registration: From global benchmark dataset to frequency-guided registration method.
  ISPRS Journal of Photogrammetry and Remote Sensing, 235:190–210, 2026.
- Benedikt Blumenstiel, Paolo Fraccaro, Valerio Marsocci, Johannes Jakubik, Stefano Maurogiovanni, Mikolaj
  Czerkawski, Rocco Sedona, Gabriele Cavallaro, Thomas Brunschwiler, Juan Bernabe Moreno, et al. TerraMesh:
  A planetary mosaic of multimodal earth observation data. In Proceedings of the IEEE/CVF Conference on
  Computer Vision and Pattern Recognition, pp. 2419–2427, 2025.
- Solène Debuysère, Nicolas Trouvé, Nathan Letheule, Elise Colin, and Georgia Channing. SARLO-80: Worldwide
  slant SAR language optic dataset 80cm. arXiv preprint arXiv:2606.20523, 2026.
- Danxu Liu, Di Wang, Hebaixu Wang, Haoyang Chen, Wentao Jiang, Yilin Cheng, Haonan Guo, Wei Cui, and Jing
  Zhang. SARMAE: Masked autoencoder for SAR representation learning. In Proceedings of the IEEE/CVF
  Conference on Computer Vision and Pattern Recognition, pp. 6496–6507, 2026.
- Yibin Ye, Xichao Teng, Hongrui Yang, Shuo Chen, Yuli Sun, Yijie Bian, Tao Tan, Zhang Li, and Qifeng Yu.
  3MOS: a multi-source, multi-resolution, and multi-scene optical-SAR dataset with insights for multi-modal
  image matching. Visual Intelligence, 3(1):19, 2025.
