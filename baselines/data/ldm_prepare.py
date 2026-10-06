#!/usr/bin/env python
"""Build the data layouts read by BBDM, cBBDM, ControlNet, SD2.1 FT and C-DiffSET.

usage: python baselines/data/ldm_prepare.py <qxs-saropt|sar2opt|sar2eo|spacenet6> [...]
env:   DATA_ROOT, WORK_DIR

Writes $WORK_DIR/ldm_data/<dataset>/ (A = SAR, B = EO):
  AB/{trainA,trainB,testA,testB}/   links to the images of the split lists
                                    (sar2opt: AB512/, centre 512 x 512 crops written as PNG)
  bbdm/{train,val,test}/{A,B}       BBDM / cBBDM tree
  controlnet/train/                 imagefolder for ControlNet: EO links + metadata.jsonl
  cdiffset/{train,test}.txt         list files for SD2.1 FT / C-DiffSET
  test_sar/                         SAR inputs of the test scripts
SAR2EO is tested on the first 4,000 test pairs (splits/sar2eo/test_first4000.txt).
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

DATASETS = ('qxs-saropt', 'sar2opt', 'sar2eo', 'spacenet6')
SPLITS = Path(__file__).resolve().parents[2] / 'splits'
PROMPT = 'electro-optical image'
# BBDM val/ subset: (source split, first n sorted names; None = the whole split)
VAL = {'qxs-saropt': ('test', 512), 'sar2opt': ('test', 64), 'sar2eo': ('test', None), 'spacenet6': ('train', 64)}


def pairs(dataset, root, list_name):
    """[(file name, sar path, eo path)] of one split list, in list order."""
    split = 'train' if list_name == 'train.txt' else 'test'
    out = []
    for eo in (SPLITS / dataset / list_name).read_text().split():
        d, base = eo.rsplit('/', 1)
        if dataset == 'spacenet6':
            base = Path(base).stem.replace('_PS-RGB', '') + '.png'
            out.append((base, root / 'chips' / f'{split}A' / base, root / 'chips' / f'{split}B' / base))
            continue
        sar = {'qxs-saropt': f'sar_256_oc_0.2/{base}',
               'sar2opt': f'{d[:-1]}A/{base}',
               'sar2eo': eo.replace('/EO/', '/SAR/', 1)}[dataset]
        out.append((base, root / sar, root / eo))
    return out


def link(target, path):
    if not os.path.lexists(path):
        os.symlink(target, path)


def link_pairs(rows, dir_a, dir_b):
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)
    for name, sar, eo in rows:
        for src, d in ((sar, dir_a), (eo, dir_b)):
            if not src.is_file():
                sys.exit(f'missing input: {src}')
            link(src, d / name)


def crop512(job):
    src, dst = job
    if dst.exists():
        return
    im = Image.open(src)
    w, h = im.size
    left, top = (w - 512) // 2, (h - 512) // 2
    tmp = dst.with_name(dst.name + '.tmp')
    im.crop((left, top, left + 512, top + 512)).save(tmp, format='PNG')
    os.replace(tmp, dst)


def build(dataset, data_root, work_dir):
    root = Path(os.path.abspath(data_root)) / dataset
    out = Path(os.path.abspath(work_dir)) / 'ldm_data' / dataset
    rows = {s: pairs(dataset, root, f'{s}.txt') for s in ('train', 'test')}

    if dataset == 'sar2opt':
        view, jobs = out / 'AB512', []
        for split, items in rows.items():
            rows[split] = [(Path(n).stem + '.png', sar, eo) for n, sar, eo in items]
            for side, idx in (('A', 1), ('B', 2)):
                (view / f'{split}{side}').mkdir(parents=True, exist_ok=True)
                jobs += [(src[idx], view / f'{split}{side}' / dst[0]) for src, dst in zip(items, rows[split])]
        with Pool(8) as pool:
            pool.map(crop512, jobs, chunksize=32)
    else:
        view = out / 'AB'
        for split, items in rows.items():
            link_pairs(items, view / f'{split}A', view / f'{split}B')
    names = {s: sorted(r[0] for r in rows[s]) for s in rows}

    test_view = view
    if dataset == 'sar2eo':
        test_view = out / 'AB4000'
        link_pairs(pairs(dataset, root, 'test_first4000.txt'), test_view / 'testA', test_view / 'testB')

    bbdm = out / 'bbdm'
    for split, src in (('train', view), ('test', test_view)):
        (bbdm / split).mkdir(parents=True, exist_ok=True)
        for side in 'AB':
            link(src / f'{split}{side}', bbdm / split / side)
    val_split, n_val = VAL[dataset]
    if n_val is None:
        (bbdm / 'val').mkdir(parents=True, exist_ok=True)
        for side in 'AB':
            link(view / f'{val_split}{side}', bbdm / 'val' / side)
    else:
        for side in 'AB':
            (bbdm / 'val' / side).mkdir(parents=True, exist_ok=True)
            for name in names[val_split][:n_val]:
                link(view / f'{val_split}{side}' / name, bbdm / 'val' / side / name)

    train_dir = out / 'controlnet' / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    with open(train_dir / 'metadata.jsonl', 'w') as meta:
        for name in names['train']:
            link(view / 'trainB' / name, train_dir / name)
            meta.write(json.dumps({'file_name': name, 'text': PROMPT,
                                   'conditioning_image': str(view / 'trainA' / name)}) + '\n')

    lists = out / 'cdiffset'
    lists.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'test'):
        if dataset in ('qxs-saropt', 'sar2opt'):
            text = (SPLITS / dataset / f'{split}.txt').read_text()
        else:
            text = ''.join(f'{split}B/{name}\n' for name in names[split])
        (lists / f'{split}.txt').write_text(text)

    link(test_view / 'testA', out / 'test_sar')
    print(f'{dataset}: layouts ready in {out}')


def main():
    if len(sys.argv) < 2 or any(d not in DATASETS for d in sys.argv[1:]):
        sys.exit(f'usage: python {sys.argv[0]} <{"|".join(DATASETS)}> [...]')
    for env in ('DATA_ROOT', 'WORK_DIR'):
        if env not in os.environ:
            sys.exit(f'set ${env}')
    for dataset in sys.argv[1:]:
        build(dataset, os.environ['DATA_ROOT'], os.environ['WORK_DIR'])


if __name__ == '__main__':
    main()
