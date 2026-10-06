"""Train the guide SegNet used by Seg-CycleGAN on the SpaceNet6 building masks.

usage: python baselines/seg-cyclegan/train_segnet.py --data-dir $WORK_DIR/gan_data/spacenet6/AB --out segnet_guide.pth
env:   BASELINES_ROOT (models/SegNet.py of baselines/seg-cyclegan/fetch.sh)

SegNet(3, 2) with its VGG16-BN encoder initialised from the torchvision ImageNet weights, trained on
the EO train tiles (read as BGR, resized to 224, scaled to [0, 1]) against SegMask/ with class-weighted
cross-entropy (0.143 background, 0.857 building), Adam (lr 1e-3), batch 16, 50 epochs. The last 128
train tiles in sorted order are held out; the checkpoint with the best holdout mIoU is kept.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.environ['BASELINES_ROOT'], 'seg-cyclegan'))
from models.SegNet import SegNet  # noqa: E402

VGG_URL = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'


class BuildingSet(torch.utils.data.Dataset):
    def __init__(self, root, names):
        self.root, self.names = root, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = cv.resize(cv.imread(str(self.root / 'trainB' / n)), (224, 224)) / 255.0
        lab = cv.imread(str(self.root / 'SegMask' / n), 0)
        lab = cv.resize(lab, (224, 224), interpolation=cv.INTER_NEAREST)
        return (torch.tensor(img, dtype=torch.float32).permute(2, 0, 1),
                torch.tensor(lab == 255, dtype=torch.long))


@torch.no_grad()
def miou(model, loader, dev):
    """-> (mean IoU over background and building, building IoU)."""
    model.eval()
    inter, union = np.zeros(2), np.zeros(2)
    for x, y in loader:
        p = model(x.to(dev)).argmax(1).cpu().numpy()
        y = y.numpy()
        for c in (0, 1):
            inter[c] += ((p == c) & (y == c)).sum()
            union[c] += ((p == c) | (y == c)).sum()
    model.train()
    return float(np.mean(inter / np.maximum(union, 1))), float(inter[1] / max(union[1], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True, help='directory with trainA/, trainB/ and SegMask/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = a.device

    names = sorted(p.name for p in (a.data_dir / 'trainA').glob('*.png'))
    tr, ho = names[:-128], names[-128:]
    print(f'train {len(tr)} / holdout {len(ho)}')
    ldr = torch.utils.data.DataLoader(BuildingSet(a.data_dir, tr), batch_size=a.bs,
                                      shuffle=True, num_workers=4, drop_last=True)
    hld = torch.utils.data.DataLoader(BuildingSet(a.data_dir, ho), batch_size=a.bs, num_workers=2)

    model = SegNet(input_channels=3, output_channels=2)
    torch.hub.load_state_dict_from_url(VGG_URL, progress=False)
    model.load_weights(os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(VGG_URL)))
    model.to(dev).train()

    ce = nn.CrossEntropyLoss(weight=torch.tensor([0.143, 0.857]).to(dev))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = -1.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        tot = 0.0
        for x, y in ldr:
            opt.zero_grad()
            loss = ce(model(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            tot += loss.item()
        m, fg = miou(model, hld, dev)
        print(f'ep {ep:3d} loss {tot / len(ldr):.4f} holdout mIoU {m:.4f} building-IoU {fg:.4f}', flush=True)
        if m > best:
            best = m
            torch.save(model.state_dict(), a.out)
    print(f'best holdout mIoU {best:.4f} -> {a.out}')


if __name__ == '__main__':
    main()
