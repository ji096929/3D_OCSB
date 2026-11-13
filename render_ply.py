"""把导出的 PLY 渲染成 PNG（3 视角，区域点高亮、背景淡化）。

用法：python render_ply.py [--only counter_007,cabinet_009]
输出：3D_OCSB/ply_png/<name>_<tag>.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLY_DIR = os.path.join(HERE, 'ply')
OUT_DIR = os.path.join(HERE, 'ply_png')
VIEWS = ((22, -55, 'front'), (22, 35, 'side'), (75, -90, 'top'))
MAX_PTS = 12000


def read_ply(path):
    with open(path, 'r') as f:
        n = 0
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            if line.strip() == 'end_header':
                break
        data = np.loadtxt(f, max_rows=n)
    return data[:, 0:3], data[:, 3:6]


def show(ax, xyz, rgb, elev, azim, title):
    is_bpos = (rgb[:, 1] > 180) & (rgb[:, 0] < 80) & (rgb[:, 2] < 80)
    is_bneg = (rgb[:, 0] > 180) & (rgb[:, 1] < 80) & (rgb[:, 2] < 80)
    is_cpos = (rgb[:, 0] > 180) & (rgb[:, 1] > 180) & (rgb[:, 2] < 80)
    bg = ~(is_bpos | is_bneg | is_cpos)

    if bg.sum():
        ax.scatter(xyz[bg, 0], xyz[bg, 1], xyz[bg, 2], c='#c8c8c8',
                   s=0.25, alpha=0.35, linewidths=0)
    if is_cpos.sum():
        ax.scatter(xyz[is_cpos, 0], xyz[is_cpos, 1], xyz[is_cpos, 2],
                   c='#ffd400', s=6, linewidths=0, label='C_pos')
    if is_bpos.sum():
        ax.scatter(xyz[is_bpos, 0], xyz[is_bpos, 1], xyz[is_bpos, 2],
                   c='#00c000', s=6, linewidths=0, label='B_pos')
    if is_bneg.sum():
        ax.scatter(xyz[is_bneg, 0], xyz[is_bneg, 1], xyz[is_bneg, 2],
                   c='#e00000', s=6, linewidths=0, label='B_neg')
    ax.view_init(elev, azim)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.set_box_aspect((max(np.ptp(xyz[:, 0]), 1e-3),
                       max(np.ptp(xyz[:, 1]), 1e-3),
                       max(np.ptp(xyz[:, 2]), 1e-3)))


def render(name, xyz, rgb):
    rng = np.random.RandomState(0)
    if len(xyz) > MAX_PTS:
        idx = rng.choice(len(xyz), MAX_PTS, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    fig = plt.figure(figsize=(14, 4.4))
    for i, (elev, azim, vname) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        show(ax, xyz, rgb, elev, azim, '%s (%s)' % (name, vname))
        if i == 2:
            ax.legend(loc='upper right', fontsize=7, markerscale=2)
    plt.tight_layout()
    dst = os.path.join(OUT_DIR, '%s.png' % name)
    plt.savefig(dst, dpi=120, bbox_inches='tight')
    plt.close()
    print('  ->', dst, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(PLY_DIR, '*_tag.ply')))
    if args.only:
        keep = [x.strip() for x in args.only.split(',')]
        files = [f for f in files
                 if any(k in os.path.basename(f) for k in keep)]
    print('tag files:', len(files), flush=True)
    for p in files:
        base = os.path.basename(p)[:-4]
        xyz, rgb = read_ply(p)
        print('rendering', base, len(xyz), flush=True)
        render(base, xyz, rgb)


if __name__ == '__main__':
    main()
