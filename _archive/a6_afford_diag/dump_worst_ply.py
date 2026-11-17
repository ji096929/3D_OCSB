"""把 worst_samples.npz 导出为 ply（GT 绿 / 预测红 / 重叠黄 / 其余灰）。

用法：
  python dump_worst_ply.py --npz /data01/jh/afford_diag/full_k5/worst_samples.npz \
      --out /data01/jh/afford_diag/ply
"""
import argparse
import json
import os

import numpy as np


def write_ply(path, xyz, rgb):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write('element vertex %d\n' % len(xyz))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyz)):
            f.write('%.5f %.5f %.5f %d %d %d\n'
                    % (xyz[i, 0], xyz[i, 1], xyz[i, 2],
                       int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.npz)
    meta = json.loads(z['meta'].tobytes().decode('utf-8'))

    index = []
    for m in meta:
        idx = m['idx']
        coord = z['coord_%d' % idx]
        gt = z['gt_%d' % idx]        # [18, N]
        pred = z['pred_%d' % idx]    # [18, N]
        best = None
        for an in m['valid_affs']:
            a = None
            # 需要 affordance 顺序；用 meta 里没有的，改用外部顺序
            # 这里退化为：按行号遍历，由调用方保证 valid_affs 与行索引一致
            pass
        index.append(dict(idx=idx, cls=m['cls'], shape_id=m['shape_id'],
                          iou=m['iou'], valid_affs=m['valid_affs']))

    # 由于 npz 不带 affordance 名称顺序，这里按行导出全部 18 行中 IoU 最低的一行
    for m in meta:
        idx = m['idx']
        coord = z['coord_%d' % idx]
        gt = z['gt_%d' % idx]
        pred = z['pred_%d' % idx]
        best_a, best_iou = -1, 2.0
        for a in range(gt.shape[0]):
            g = gt[a] > 0.5
            if g.sum() == 0:
                continue
            p = pred[a] > 0.5
            tp = (p & g).sum()
            iou = tp / max((p | g).sum(), 1)
            if iou < best_iou:
                best_iou, best_a = iou, a
        if best_a < 0:
            continue
        g = gt[best_a] > 0.5
        p = pred[best_a] > 0.5
        rgb = np.full((len(coord), 3), 60, np.uint8)
        rgb[g] = [0, 220, 0]
        rgb[p] = [230, 0, 0]
        rgb[g & p] = [255, 210, 0]
        name = 'worst_%03d_%s_a%02d_iou%.3f.ply' % (idx, m['cls'], best_a, best_iou)
        write_ply(os.path.join(args.out, name), coord, rgb)
        print('%-46s pts=%d gt=%d pred=%d iou=%.4f'
              % (name, len(coord), int(g.sum()), int(p.sum()), best_iou))
        index[-1].update(aff_row=best_a, points=int(len(coord)),
                         gt_pos=int(g.sum()), pred_pos=int(p.sum()),
                         best_iou=float(best_iou))

    json.dump(index, open(os.path.join(args.out, 'index.json'), 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
