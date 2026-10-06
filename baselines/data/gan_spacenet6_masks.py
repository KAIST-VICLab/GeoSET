"""Rasterize SpaceNet6 building footprints into the SegMask/ directory read by Seg-CycleGAN.

usage: python baselines/data/gan_spacenet6_masks.py
env:   DATA_ROOT, WORK_DIR (run gan_prepare.py spacenet6 first); requires rasterio

For every train tile, the footprints in geojson_buildings (tile CRS) are burned in as 255 on the
900 x 900 PS-RGB grid of the tile and resized to 256 x 256 (nearest), aligned with the chips.
Output: $WORK_DIR/gan_data/spacenet6/AB/SegMask/<chip name>.png (0 = background, 255 = building).
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gan_prepare import SPLITS, read_list  # noqa: E402


def building_mask(tif, geojson, size=256):
    with rasterio.open(tif) as src:
        transform, shape = src.transform, (src.height, src.width)
    shapes = []
    if geojson.exists():
        shapes = [(f['geometry'], 255) for f in json.loads(geojson.read_text())['features'] if f.get('geometry')]
    if shapes:
        m = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype=np.uint8)
    else:
        m = np.zeros(shape, np.uint8)
    return Image.fromarray(m).resize((size, size), Image.NEAREST)


def build(data_root, work_dir, splits_dir=SPLITS):
    root = Path(os.path.abspath(data_root)) / 'spacenet6'
    dst = Path(os.path.abspath(work_dir)) / 'gan_data' / 'spacenet6' / 'AB' / 'SegMask'
    if dst.exists():
        print(f'{dst} exists, skipping')
        return
    tmp = dst.with_name('SegMask.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n_empty = 0
    for eo_rel in read_list(Path(splits_dir) / 'spacenet6' / 'train.txt'):
        tif = root / eo_rel
        gj = tif.parent.parent / 'geojson_buildings' / tif.name.replace('_PS-RGB_', '_Buildings_')
        m = building_mask(tif, gj.with_suffix('.geojson'))
        n_empty += not m.getbbox()
        m.save(tmp / (tif.stem.replace('_PS-RGB', '') + '.png'))
    tmp.rename(dst)
    print(f'{len(os.listdir(dst))} masks ({n_empty} without buildings) in {dst}')


if __name__ == '__main__':
    build(os.environ['DATA_ROOT'], os.environ['WORK_DIR'])
