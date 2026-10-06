"""Build the data views read by the GAN-family baselines, for one dataset.

usage: python baselines/data/gan_prepare.py <qxs-saropt|sar2opt|sar2eo|spacenet6>
env:   DATA_ROOT, WORK_DIR, BASELINES_ROOT (combine_A_and_B.py of baselines/pix2pix/fetch.sh)

Writes $WORK_DIR/gan_data/<dataset>/ (A = SAR, B = EO):
  AB/{trainA,trainB,testA,testB}/        links to the images of the split lists
  combined/{train,test}/                 side-by-side A|B images for pix2pix
  p2phd/{train_A,train_B,test_A,test_B}  directory links for pix2pixHD
  sar2opt only: test512/{testA,testB}/ and combined_test512/test/, the centre 512 x 512 test crops
SAR2EO uses the first 4,000 test pairs (splits/sar2eo/test_first4000.txt).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

SPLITS = Path(__file__).resolve().parents[2] / 'splits'
DATASETS = ('qxs-saropt', 'sar2opt', 'sar2eo', 'spacenet6')


def read_list(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def pairs(dataset, root, splits_dir, split):
    """-> [(name, sar_path, eo_path)] for one split, in split-list order."""
    lst = 'test_first4000.txt' if (dataset, split) == ('sar2eo', 'test') else f'{split}.txt'
    out = []
    for eo_rel in read_list(splits_dir / dataset / lst):
        d, base = eo_rel.rsplit('/', 1)
        if dataset == 'spacenet6':
            name = Path(base).stem.replace('_PS-RGB', '') + '.png'
            out.append((name, root / 'chips' / f'{split}A' / name, root / 'chips' / f'{split}B' / name))
            continue
        sar_rel = {'qxs-saropt': f'sar_256_oc_0.2/{base}',
                   'sar2opt': f'{d[:-1]}A/{base}',
                   'sar2eo': eo_rel.replace('/EO/', '/SAR/', 1)}[dataset]
        out.append((base, root / sar_rel, root / eo_rel))
    return out


def fresh(dst):
    """Temporary build dir for dst, or None if dst is already complete."""
    if dst.exists():
        print(f'{dst} exists, skipping')
        return None
    tmp = dst.with_name(dst.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    return tmp


def build_ab(dst, items):
    tmp = fresh(dst)
    if tmp is None:
        return
    for split, rows in items.items():
        for side in 'AB':
            (tmp / f'{split}{side}').mkdir()
        for name, sar, eo in rows:
            for side, src in (('A', sar), ('B', eo)):
                if not src.is_file():
                    sys.exit(f'missing input: {src}')
                os.symlink(src, tmp / f'{split}{side}' / name)
    tmp.rename(dst)


def build_combined(dst, ab, combine_script):
    tmp = fresh(dst)
    if tmp is None:
        return
    stage = dst.with_name(dst.name + '.stage')
    shutil.rmtree(stage, ignore_errors=True)
    for side in 'AB':
        (stage / side).mkdir(parents=True)
        for split in ('train', 'test'):
            os.symlink(ab / f'{split}{side}', stage / side / split)
    subprocess.run([sys.executable, str(combine_script), '--fold_A', str(stage / 'A'),
                    '--fold_B', str(stage / 'B'), '--fold_AB', str(tmp)], check=True)
    for split in ('train', 'test'):
        want, got = len(os.listdir(ab / f'{split}A')), len(os.listdir(tmp / split))
        if got != want:
            sys.exit(f'combine_A_and_B.py wrote {got} of {want} {split} images')
    shutil.rmtree(stage)
    tmp.rename(dst)


def center512(img, s=512):
    w, h = img.size
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def build_test512(dst, ab):
    tmp = fresh(dst)
    if tmp is None:
        return
    for side in ('testA', 'testB'):
        (tmp / side).mkdir()
        for p in sorted((ab / side).iterdir()):
            center512(Image.open(p)).save(tmp / side / f'{p.stem}.png')
    tmp.rename(dst)


def build_combined_test512(dst, combined):
    tmp = fresh(dst)
    if tmp is None:
        return
    (tmp / 'test').mkdir()
    for p in sorted((combined / 'test').iterdir()):
        im = Image.open(p)
        w, h = im.size
        canvas = Image.new(im.mode, (1024, 512))
        canvas.paste(center512(im.crop((0, 0, w // 2, h))), (0, 0))
        canvas.paste(center512(im.crop((w // 2, 0, w, h))), (512, 0))
        canvas.save(tmp / 'test' / f'{p.stem}.png')
    tmp.rename(dst)


def build_p2phd(dst, train_dir, test_dir):
    tmp = fresh(dst)
    if tmp is None:
        return
    for split, src in (('train', train_dir), ('test', test_dir)):
        for side in 'AB':
            os.symlink(src / f'{split}{side}', tmp / f'{split}_{side}')
    tmp.rename(dst)


def build(dataset, data_root, work_dir, combine_script, splits_dir=SPLITS):
    root = Path(os.path.abspath(data_root)) / dataset
    out = Path(os.path.abspath(work_dir)) / 'gan_data' / dataset
    out.mkdir(parents=True, exist_ok=True)
    ab = out / 'AB'
    build_ab(ab, {split: pairs(dataset, root, Path(splits_dir), split) for split in ('train', 'test')})
    build_combined(out / 'combined', ab, combine_script)
    test_dir = ab
    if dataset == 'sar2opt':
        test_dir = out / 'test512'
        build_test512(test_dir, ab)
        build_combined_test512(out / 'combined_test512', out / 'combined')
    build_p2phd(out / 'p2phd', ab, test_dir)
    print(f'{dataset}: views ready in {out}')


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in DATASETS:
        sys.exit(f'usage: python {sys.argv[0]} <{"|".join(DATASETS)}>')
    combine_script = Path(os.environ['BASELINES_ROOT']) / 'pix2pix' / 'datasets' / 'combine_A_and_B.py'
    build(sys.argv[1], os.environ['DATA_ROOT'], os.environ['WORK_DIR'], combine_script)


if __name__ == '__main__':
    main()
