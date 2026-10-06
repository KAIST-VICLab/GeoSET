#!/usr/bin/env python
"""Data layouts for the diffusion baselines (DDPM (SR3), E3Diff, CondDiff, ResShift, HI-Diff).

    python baselines/data/diff_data.py paired   <dataset>   # ResShift, HI-Diff
    python baselines/data/diff_data.py conddiff <dataset>   # CondDiff
    python baselines/data/diff_data.py e3diff   <dataset>   # DDPM (SR3), E3Diff

Reads $DATA_ROOT and splits/<dataset>/*.txt and writes $WORK_DIR/diff_data/<dataset>/<layout>/.
Images are symlinked, except the SAR2Opt 512x512 centre crops (written as PNG) and the
E3Diff SAR-PPB / SAR-canny conditions (computed on the GPU when one is available).
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

DATASETS = ('qxs-saropt', 'sar2opt', 'sar2eo', 'spacenet6')
SPLITS = Path(__file__).resolve().parents[2] / 'splits'


def env(name):
    if name not in os.environ:
        raise SystemExit(f'set ${name}')
    return Path(os.environ[name])


def pairs(dataset, listname):
    """[(name, sar_path, eo_path)] in list order; name is the file name used in the derived trees."""
    root = env('DATA_ROOT') / dataset
    out = []
    for eo in (SPLITS / dataset / f'{listname}.txt').read_text().split():
        if dataset == 'qxs-saropt':
            out.append((Path(eo).name, root / eo.replace('opt_256_oc_0.2/', 'sar_256_oc_0.2/', 1), root / eo))
        elif dataset == 'sar2opt':
            folder, fname = eo.split('/', 1)
            out.append((fname, root / (folder[:-1] + 'A') / fname, root / eo))
        elif dataset == 'sar2eo':
            out.append((Path(eo).name, root / eo.replace('/EO/', '/SAR/', 1), root / eo))
        else:
            split = listname.split('_')[0]
            name = Path(eo).stem.replace('_PS-RGB', '') + '.png'
            out.append((name, root / 'chips' / f'{split}A' / name, root / 'chips' / f'{split}B' / name))
    return out


def link(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if os.readlink(path) == str(target):
            return
        path.unlink()
    os.symlink(target, path)


def link_pairs(items, dir_a, dir_b):
    for name, sar, eo in items:
        link(sar, dir_a / name)
        link(eo, dir_b / name)


def _crop(job):
    src, dst = job
    if dst.exists():
        return
    im = Image.open(src)
    w, h = im.size
    left, top = (w - 512) // 2, (h - 512) // 2
    tmp = dst.with_name('.tmp_' + dst.name)
    im.crop((left, top, left + 512, top + 512)).save(tmp)
    os.replace(tmp, dst)


def crop512(items, split):
    """SAR2Opt: deterministic 512x512 centre crops of the 600x600 tiles, as PNG."""
    root = env('WORK_DIR') / 'diff_data' / 'sar2opt' / 'crop512'
    out, jobs = [], []
    for name, sar, eo in items:
        name = Path(name).stem + '.png'
        a, b = root / f'{split}A' / name, root / f'{split}B' / name
        a.parent.mkdir(parents=True, exist_ok=True)
        b.parent.mkdir(parents=True, exist_ok=True)
        jobs += [(sar, a), (eo, b)]
        out.append((name, a, b))
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(_crop, jobs))
    return out


def paired(dataset):
    """{train,test}{A,B} (A = SAR) and the HI-Diff monitoring subset val{A,B}; SAR2EO also gets
    meta_train.txt (HI-Diff) and test_first4000A (ResShift inference)."""
    out = env('WORK_DIR') / 'diff_data' / dataset / 'paired'
    for split in ('train', 'test'):
        items = pairs(dataset, split)
        if dataset == 'sar2opt' and split == 'test':
            items = crop512(items, split)
        link_pairs(items, out / f'{split}A', out / f'{split}B')
    test = sorted(items)
    if dataset == 'sar2eo':
        names = sorted(n for n, _, _ in pairs(dataset, 'train'))
        (out / 'meta_train.txt').write_text(''.join(n + '\n' for n in names))
        for name, sar, _ in pairs(dataset, 'test_first4000'):
            link(sar, out / 'test_first4000A' / name)
    if dataset == 'qxs-saropt':
        val = sorted(test, key=lambda t: int(Path(t[0]).stem))[:100]
    elif dataset == 'sar2opt':
        val = test[:64]
    else:
        val = test[::max(1, len(test) // 100)][:100]
    link_pairs(val, out / 'valA', out / 'valB')
    print(f'[paired] {dataset}: {out}')


def conddiff(dataset):
    """{train,test}/{sar,opt}/<6-digit id> plus manifest_{train,test}.json (id -> name)."""
    out = env('WORK_DIR') / 'diff_data' / dataset / 'conddiff'
    for split in ('train', 'test'):
        items = pairs(dataset, 'test_first4000' if dataset == 'sar2eo' and split == 'test' else split)
        if dataset == 'sar2opt':
            items = crop512(items, split)
        items.sort(key=lambda t: Path(t[0]).stem)
        manifest = {}
        for i, (name, sar, eo) in enumerate(items):
            pid = f'{i:06d}'
            manifest[pid] = Path(name).stem
            link(sar, out / split / 'sar' / (pid + Path(sar).suffix.lower()))
            link(eo, out / split / 'opt' / (pid + Path(eo).suffix.lower()))
        (out / f'manifest_{split}.json').write_text(json.dumps(manifest))
        print(f'[conddiff] {dataset}/{split}: {len(items)} pairs')


# FAST_PPB (Deledalle et al., 2009) with the E3Diff parameters, then Canny on the 8-bit result.
P, W, H_SMOOTH = 3, 10, 0.5
PAD = W + P + 1
K = 2 * P + 1


def ppb(xpad, h, w):
    """Non-iterative PPB on a batch of symmetric-padded images, shape (B, h + 2*PAD, w + 2*PAD)."""
    import torch
    n, m = h + 2 * P + 1, w + 2 * P + 1
    out = torch.zeros((xpad.shape[0], h, w), dtype=xpad.dtype, device=xpad.device)
    wmax = torch.zeros_like(out)
    wsum = torch.zeros_like(out)
    a = xpad[:, W:W + n, W:W + m]
    for dx in range(-W, W + 1):
        for dy in range(-W, W + 1):
            if dx == 0 and dy == 0:
                continue
            b = xpad[:, W + dx:W + dx + n, W + dy:W + dy + m]
            isd = torch.log(a / b + b / a).cumsum(1).cumsum(2)
            ssd = isd[:, :h, :w] + isd[:, K:, K:] - isd[:, :h, K:] - isd[:, K:, :w]
            wgt = torch.exp(-ssd / H_SMOOTH)
            v = xpad[:, PAD + dx:PAD + dx + h, PAD + dy:PAD + dy + w]
            out += wgt * v ** 2
            wmax = torch.maximum(wmax, wgt)
            wsum += wgt
    c = xpad[:, PAD:PAD + h, PAD:PAD + w]
    return torch.sqrt((out + wmax * c ** 2) / (wsum + wmax))


def _load(path):
    import cv2
    import numpy as np
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(f'cannot read {path}')
    im = im.astype(np.float64) / 255.0
    pos = im[im > 0]
    if pos.size == 0:
        return None, im.shape
    im[im == 0] = pos.min()
    return np.pad(im, PAD, mode='symmetric'), im.shape


def _save(job):
    import cv2
    name, ppb_u8, ppb_dir, canny_dir = job
    canny = cv2.Canny(ppb_u8, 50, 150, L2gradient=True)
    if not (cv2.imwrite(str(ppb_dir / name), ppb_u8) and cv2.imwrite(str(canny_dir / name), canny)):
        raise IOError(f'write failed for {name}')


def conditions(sar_dir, names, ppb_dir, canny_dir, batch):
    import numpy as np
    import torch
    ppb_dir.mkdir(parents=True, exist_ok=True)
    canny_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not ((ppb_dir / n).exists() and (canny_dir / n).exists())]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with ThreadPoolExecutor(16) as pool:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            keep = []
            for n, (x, shape) in zip(chunk, pool.map(lambda n: _load(sar_dir / n), chunk)):
                if x is None:
                    _save((n, np.zeros(shape, np.uint8), ppb_dir, canny_dir))
                else:
                    keep.append((n, x))
            if not keep:
                continue
            hh, ww = keep[0][1].shape
            out = ppb(torch.from_numpy(np.stack([x for _, x in keep])).to(dev), hh - 2 * PAD, ww - 2 * PAD)
            mn = out.amin(dim=(1, 2), keepdim=True)
            mx = out.amax(dim=(1, 2), keepdim=True)
            u8 = torch.clamp(torch.round((out - mn) / (mx - mn) * 255.0), 0, 255).to(torch.uint8).cpu().numpy()
            list(pool.map(_save, [(n, u8[j], ppb_dir, canny_dir) for j, (n, _) in enumerate(keep)]))
            print(f'[e3diff] {ppb_dir.parent.name}: {min(i + batch, len(todo))}/{len(todo)}', flush=True)


def e3diff(dataset):
    """{train,val}/{SAR,EO,SAR-PPB,SAR-canny}/<name>; val is the test split (SAR2EO: its first 4,000)."""
    out = env('WORK_DIR') / 'diff_data' / dataset / 'e3diff'
    for split, sub in (('train', 'train'), ('test', 'val')):
        items = pairs(dataset, 'test_first4000' if dataset == 'sar2eo' and split == 'test' else split)
        if dataset == 'sar2opt':
            items = crop512(items, split)
        link_pairs(items, out / sub / 'SAR', out / sub / 'EO')
        conditions(out / sub / 'SAR', sorted(n for n, _, _ in items),
                   out / sub / 'SAR-PPB', out / sub / 'SAR-canny', 32 if dataset == 'sar2opt' else 128)
    print(f'[e3diff] {dataset}: {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('layout', choices=('paired', 'conddiff', 'e3diff'))
    ap.add_argument('dataset', choices=DATASETS)
    args = ap.parse_args()
    globals()[args.layout](args.dataset)
