# Pair filtering and source sampling

Stage 2 of GeoSET is pretrained on paired SAR–EO observations from five dataset families that form six
streams (GUSO, TerraMesh, SARLO-80, SAR-1M, and 3MOS split into `3mos_mr` and `3mos_hr`; see
[CORPUS.md](CORPUS.md)). This page describes how the pairs were selected and weighted (paper Section 4.1
and Table 1), the keep tables that define the selected pairs, and how to regenerate them.

## Eligibility

Stage 2 uses only paired SAR–EO observations. Source-specific eligibility criteria are applied first:

- **TerraMesh**: pairs with reported zero cloud cover and an acquisition-time difference of at most one
  day are retained, yielding 1,760,233 eligible pairs (21.5%) from 8,194,048 SAR observations.
- **SAR-1M**: the samples with an available EO counterpart (the entries of `paired.json`) are retained,
  out of 1,130,379 SAR observations.
- **GUSO, SARLO-80, 3MOS**: every SAR sample with its EO counterpart (GUSO: 589,137 of 589,143 tiles;
  SARLO-80: 87,869 of 87,870 samples, one SAR file being incomplete; 3MOS: all 113,074).

Together, these criteria yield **3,281,393** eligible pairs.

## Quality filtering

Eligible pairs exhibiting any of six defects are removed:

| category | pairs | reason code |
|---|---:|---|
| featureless content | 51,515 | `featureless` |
| cloud contamination | 18,722 | `cloud` |
| SAR–EO misregistration | 6,413 | `misregistration` |
| compression artifacts | 582 | `compression` |
| missing tiles | 270 | `missing_tile` |
| dead/no-data regions | 161 | `no_data` |

The categories are non-exclusive, so their counts do not sum to the number of removed pairs. Removing
the union leaves **3,204,744** pairs.

The `featureless`, `cloud`, `compression` and `no_data` rules are computed from the EO image alone, on one
256 × 256 crop per pair (the whole image for 256-px sources); `misregistration` and `missing_tile` use the
registration of the SARLO-80 optical images. With luminance `0.299 R + 0.587 G + 0.114 B` of the EO image
scaled to [0, 1]:

- `flat`: fraction of the non-overlapping 16 × 16 luminance cells whose variance is below 1e-5;
- `cloudish`: fraction of pixels whose largest RGB value, on [-1, 1], exceeds 0.4 and whose saturation
  `(max - min) / (max + 1) * 2` is below 0.15;
- `block`: mean absolute horizontal luminance step on the columns `8k + 7` minus the mean off them,
  divided by the latter;
- `lapvar`: variance of the 3 × 3 Laplacian of the luminance.

| reason code | rule | streams |
|---|---|---|
| `featureless` | `flat > 0.25` | all |
| `cloud` | `cloudish > 0.3` | `sar1m`, `3mos_mr`, `3mos_hr` |
| `misregistration` | SAR-to-optical registration residual above 2 SAR pixels (0.8 m) | `sarlo80` |
| `compression` | `block` above the stream threshold and `lapvar` above the stream median; applied to streams whose median `block` exceeds 0.01 | `sar1m` (0.6145), `3mos_mr` (0.4069), `3mos_hr` (0.4544) |
| `missing_tile` | the optical image has map tiles that could not be retrieved | `sarlo80` |
| `no_data` | `flat > 0.5` | `3mos_mr` |

Per stream:

| stream | eligible | kept | featureless | cloud | misregistration | compression | missing_tile | no_data |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `guso` | 589,137 | 585,424 | 3,713 | | | | | |
| `terramesh` | 1,760,233 | 1,741,434 | 18,799 | | | | | |
| `sarlo80` | 87,869 | 77,932 | 3,443 | | 6,413 | | 270 | |
| `sar1m` | 731,080 | 688,458 | 24,897 | 17,903 | | 453 | | |
| `3mos_mr` | 87,293 | 85,924 | 599 | 674 | | 129 | | 161 |
| `3mos_hr` | 25,781 | 25,572 | 64 | 145 | | | | |
| total | 3,281,393 | 3,204,744 | 51,515 | 18,722 | 6,413 | 582 | 270 | 161 |

## Source sampling

Native image sizes and crop densities vary substantially across sources. Source sampling probabilities are
determined from the number of non-overlapping 256 × 256 crop equivalents. For 3MOS, the tile count is
divided by four to approximately account for the redundancy introduced by half-tile strides along both
spatial axes. This correction reduces the overrepresentation of densely overlapping tiles in the
pretraining mixture.

```
crop_equivalents = kept_pairs * (tile_px // 256)**2 // redundancy        # redundancy 4 for 3MOS, else 1
weight           = round(10000 * crop_equivalents / total)                # residual goes to the largest stream
```

Paper Table 1 (drop rates are relative to the original SAR sample counts and include both eligibility
selection and quality filtering):

| source | orig. samples | kept pairs | drop (%) | crop equivalents | mix (%) |
|---|---:|---:|---:|---:|---:|
| GUSO | 589,143 | 585,424 | 0.63 | 2,341,696 | 38.73 |
| TerraMesh | 8,194,048 | 1,741,434 | 78.75 | 1,741,434 | 28.80 |
| SARLO-80 | 87,870 | 77,932 | 11.31 | 1,246,912 | 20.62 |
| SAR-1M | 1,130,379 | 688,458 | 39.09 | 688,458 | 11.39 |
| 3MOS | 113,074 | 111,496 | 1.40 | 27,874 | 0.46 |
| total | 10,114,514 | 3,204,744 | 68.32 | 6,046,374 | 100.00 |

Mix weights per stream, in parts of 10,000:

| stream | tile px | crops per tile | redundancy | stage-2 crop equivalents | stage-2 weight | stage-1 weight |
|---|---:|---:|---:|---:|---:|---:|
| `guso` | 512 | 4 | 1 | 2,341,696 | 3872 | 1853 |
| `terramesh` | 264 | 1 | 1 | 1,741,434 | 2880 | 6444 |
| `sarlo80` | 1024 | 16 | 1 | 1,246,912 | 2062 | 1106 |
| `sar1m` | 256 | 1 | 1 | 688,458 | 1139 | 575 |
| `3mos_mr` | 256 | 1 | 4 | 21,481 | 36 | 17 |
| `3mos_hr` | 256 | 1 | 4 | 6,393 | 11 | 5 |

Stage 1 uses only SAR images; its weights apply the same rule to the SAR sample counts of each stream
(listed in `corpus/MANIFEST.json`).

## Keep tables

`corpus/keep_v1/<stream>.tsv.gz` lists every eligible pair of a stream, one row per pair:

```
sample_id <TAB> scene_id <TAB> keep <TAB> reasons
```

- `keep` is `1` for the 3,204,744 pairs used in stage 2 and `0` otherwise;
- `reasons` is the comma-separated subset of `featureless,cloud,misregistration,compression,missing_tile,no_data`
  that removed the pair (empty when kept);
- `scene_id` groups pairs of one scene, site or sensor group; the stage-2 loader draws a scene within a
  stream and then a pair within that scene;
- `sample_id` identifies the SAR sample (paths are relative to the stream's directory under `$CORPUS_ROOT`):

| stream | `sample_id` | example |
|---|---|---|
| `guso` | SAR file under `guso/` | `urban/train/Europe/231_Europe_Turkey_000656_SAR.tif` |
| `terramesh` | `<split>/<zarr member name>` | `train/majortom_train_0000001.zarr.zip` |
| `sarlo80` | SAR file under `sarlo80/` | `train_x/chunk_006/shard-00029/00065060.sar.npy` |
| `sar1m` | `sar1m/<subset>/<SAR file stem>` (subset from the file-name convention, see `prepare_sar1m.py`) | `sar1m/m4sar/67506_1` |
| `3mos_mr`, `3mos_hr` | SAR file under `3mos/` | `SEN/SEN1_region7/sar/sar_1773.jpg` |

`corpus/MANIFEST.json` records, per stream, the eligible, kept and removed counts, the counts per reason,
the crop equivalents and the stage-2 weight, the stage-1 sample counts and weights, and the SHA-256 of each
uncompressed table. The stage-2 data loader (`geoset.data.PairedCorpus`) reads the tables from
`corpus/keep_v1/` and uses only rows with `keep = 1`.

TerraMesh is addressed by member name. At revision `9f13e7172f16e7e04bae8c303fe10e32dcde845d` of the
dataset, 1,709,540 of the 1,741,434 kept TerraMesh pairs are distributed; members that are not present
are skipped by the loader.

## Regenerating the keep tables

The published tables define the pretraining set. The scripts in `scripts/corpus/` apply the same
eligibility criteria and rules to a local copy of the corpus (layout in [CORPUS.md](CORPUS.md)); they
need `numpy`, `torch`, `pillow`, `pyarrow` and, for TerraMesh, `zarr` 2.x.

```bash
W=work   # intermediate lists

# 1. pair lists: sample_id, scene_id, SAR path, EO path
python scripts/corpus/prepare_terramesh.py --root $CORPUS_ROOT/terramesh --out $W/pairs/terramesh.tsv
python scripts/corpus/prepare_sar1m.py --root $CORPUS_ROOT/sar1m --out $W/pairs/sar1m.tsv
python scripts/corpus/list_pairs.py --source guso --root $CORPUS_ROOT/guso --out-dir $W/pairs
python scripts/corpus/list_pairs.py --source sarlo80 --root $CORPUS_ROOT/sarlo80 --out-dir $W/pairs
python scripts/corpus/list_pairs.py --source 3mos --root $CORPUS_ROOT/3mos --out-dir $W/pairs   # 3mos_mr, 3mos_hr

# 2. EO scores (block, lapvar, flat, cloudish); add --device cuda to use a GPU
for s in guso terramesh sarlo80 sar1m; do
  python scripts/corpus/score_eo_iqa.py --root $CORPUS_ROOT/$s --pairs $W/pairs/$s.tsv --out $W/scores/$s.tsv
done
for s in 3mos_mr 3mos_hr; do
  python scripts/corpus/score_eo_iqa.py --root $CORPUS_ROOT/3mos --pairs $W/pairs/$s.tsv --out $W/scores/$s.tsv
done

# 3. rules -> keep tables and the stage-2 entries of MANIFEST.json
python scripts/corpus/build_keep_tables.py --pairs $W/pairs --scores $W/scores \
    --registration $W/sarlo80_registration.tsv --out $W/corpus
```

- `--registration` (SARLO-80) is a TSV with columns `sample_id`, `resid_px` and `missing_tiles`: the
  largest control-point residual of the SAR-to-optical transform in 0.8 m SAR pixels and the number of
  optical map tiles that could not be retrieved. These values come from the optical reconstruction and drive
  only the `misregistration` and `missing_tile` rules, which are not applied without them; the published
  SARLO-80 table remains the reference for these two rules.
- GUSO (512 px), TerraMesh (264 px) and SARLO-80 (1024 px) images are scored on one 256 × 256 crop whose
  position is drawn per sample, so a re-run can differ from the published tables on individual
  `featureless` decisions of these streams.
- On TerraMesh revision `9f13e7172f16e7e04bae8c303fe10e32dcde845d`, `prepare_terramesh.py` finds
  1,727,960 eligible Major TOM pairs (train 1,711,202, val 16,758), all of which are rows of the published
  table.
