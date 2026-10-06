#!/usr/bin/env python
"""E3Diff main.py with a 3-channel EO target and a 3-channel condition [PPB, canny, SAR].

E3Diff's SAR_EO loader returns a 1-channel EO target and a [PPB, canny] condition. For RGB targets
this wrapper replaces SAR2EODataset.__getitem__ in memory and then runs the repository's main.py
unchanged. Usage is that of main.py, e.g.
  E3DIFF_REPO=<E3Diff checkout> python e3diff_rgb_main.py -c cfg.json -p train -enable_wandb "" --seed 1
"""
import os
import runpy
import sys

import torch
from PIL import Image

REPO = os.environ.get('E3DIFF_REPO')
if not REPO:
    raise SystemExit('set E3DIFF_REPO to the E3Diff checkout')
sys.path.insert(0, REPO)

import data.util as Util                       # noqa: E402
from data.LRHR_dataset import SAR2EODataset    # noqa: E402


def _getitem_rgb(self, index):
    hr = self.hr_path[index]
    filename = os.path.basename(hr)
    img_EO = Image.open(hr).convert('RGB')
    img_SAR = Image.open(os.path.join(self.dataroot, 'SAR', filename)).convert('RGB')
    img_ppb = Image.open(os.path.join(self.dataroot, 'SAR-PPB', filename)).convert('RGB')
    img_canny = Image.open(os.path.join(self.dataroot, 'SAR-canny', filename)).convert('RGB')
    img_SAR, img_ppb, img_EO, img_canny = Util.transform_augment(
        [img_SAR, img_ppb, img_EO, img_canny], split=self.split, min_max=(-1, 1))
    cond = torch.cat((img_ppb[0:1], img_canny[0:1], img_SAR[0:1]), dim=0)
    return {'HR': img_EO[0:3], 'LR': img_SAR[0:3], 'SR': cond,
            'Index': index, 'filename': filename}


SAR2EODataset.__getitem__ = _getitem_rgb

_main = os.path.join(REPO, 'main.py')
sys.argv[0] = _main
runpy.run_path(_main, run_name='__main__')
