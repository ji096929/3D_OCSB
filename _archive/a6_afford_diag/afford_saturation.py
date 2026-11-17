"""3D_AffordanceNet：APGraph 输出的饱和程度（逐对 p90/p95/p99 与 GT 占比）。

用于回答：Knife-stab 式的「整片 1.0」是否在 3D_AffordanceNet 上也普遍存在。
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = '/nas_data/ljy/AffordanceNet-main'
sys.path.insert(0, ROOT)

OUT = '/data01/jh/afford_diag/afford_saturation.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from dataset.object_incremental import ObjectIncrementalDataset
    from attribute_graph.incremental import load_pair_graph

    torch.set_num_threads(4)
    man = json.loads(open(args.manifest).read())
    affs = man['affordances']
    validity = {k: set(v) for k, v in man['validity'].items()}

    pair, _, _ = load_pair_graph(args.base_dir + '/base.pt', args.checkpoint)
    pair.eval()
    ds = ObjectIncrementalDataset(args.data_root, man, 3, 'val', all_objects=True)
    print('samples', len(ds), flush=True)

    buckets = defaultdict(lambda: [[], []])
    for i in range(len(ds)):
        coord, _, target, shape_id, cls = ds[i]
        x = torch.from_numpy(coord).float().cuda().unsqueeze(0).transpose(1, 2)
        with torch.inference_mode():
            prob = torch.sigmoid(pair(x))[0].cpu().numpy()
        gt = np.asarray(target, dtype=np.float32).T
        for a, an in enumerate(affs):
            if an in validity.get(cls, set()):
                buckets[(cls, an)][0].append(prob[a])
                buckets[(cls, an)][1].append(gt[a])
        if (i + 1) % 1000 == 0:
            print('  %d / %d' % (i + 1, len(ds)), flush=True)

    rows = []
    for (cls, an), (pl, gl) in buckets.items():
        q90 = [float(np.percentile(p, 90)) for p in pl]
        q95 = [float(np.percentile(p, 95)) for p in pl]
        q99 = [float(np.percentile(p, 99)) for p in pl]
        gt_frac = [float((g > 0.5).mean()) for g in gl]
        # 高阈值下还有多少点被激活（饱和的直接后果）
        over99 = [float((p >= 0.99).mean()) for p in pl]
        rows.append(dict(cls=cls, aff=an, n=len(pl),
                         p90=float(np.mean(q90)), p95=float(np.mean(q95)),
                         p99=float(np.mean(q99)),
                         gt_frac=float(np.mean(gt_frac)),
                         frac_ge_099=float(np.mean(over99)),
                         saturated=bool(np.mean(q90) > 0.9)))

    rows.sort(key=lambda r: -r['frac_ge_099'])
    json.dump(dict(pairs=rows), open(args.out, 'w'), indent=1)

    sat = [r for r in rows if r['saturated']]
    print('\n58 个有效对里，p90 > 0.9 的（饱和对）: %d / %d' % (len(sat), len(rows)))
    print('%-18s %-12s %6s %6s %6s %8s' % ('class', 'aff', 'GT%', 'p90', 'p99', '>=0.99'))
    for r in rows[:15]:
        print('%-18s %-12s %5.1f %6.3f %6.3f %7.1f%%'
              % (r['cls'], r['aff'], 100 * r['gt_frac'], r['p90'], r['p99'], 100 * r['frac_ge_099']))
    print('\n->', args.out)


if __name__ == '__main__':
    main()
