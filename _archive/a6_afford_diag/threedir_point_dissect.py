"""逐点解剖：GT 占比、预测概率分位数、20 档阈值 IoU 曲线。

用于回答：Knife-stab 这类 AUC 很高但 aIOU 很低的组合，到底是
  (a) 预测饱和（整片 0.95-1.0，高阈值也切不细），还是
  (b) GT 极小（2%-5%），少量高分溢出就把 IoU 压死。
"""
import json
import sys
from pathlib import Path

import numpy as np

PROTO = Path('/nas_data/ljy/baseline/3DIR_Order2143_20261109/protocol/query_all.jsonl')
RUNS = {
    'APGraph': ('/nas_data/ljy/baseline/3DIR_Order2143_20261109/APGraph/V2/cluster_no_norm_no_T/K5_seed1',
                'stage4_query.npz'),
    'IAGNet': ('/nas_data/ljy/baseline/IAGNet_DINO_HardLabel_ES20/runs/3dir/'
               '3dir_swap_grasp_wrapgrasp_with_lift_pour_v1/K5_seed1',
               'evaluations/stage4_query_predictions.npz'),
}
TARGETS = [('Knife', 'stab'), ('Suitcase', 'pull'), ('Bed', 'sit'),
           ('Baseballbat', 'grasp'), ('Knife', 'grasp'),
           ('Scissors', 'grasp'), ('Bottle', 'wrapgrasp')]


def load_run(root, name):
    z = np.load(Path(root) / name, allow_pickle=True)
    ids = z['pair_ids'] if 'pair_ids' in z else None
    return z['pred'], z['target'], ids


def main():
    rows = [json.loads(l) for l in PROTO.read_text().splitlines() if l.strip()]
    order = {r['pair_id']: i for i, r in enumerate(rows)}

    for method, (root, name) in RUNS.items():
        pred, target, ids = load_run(root, name)
        print('\n' + '=' * 78)
        print('=== %s ===' % method)
        for obj, act in TARGETS:
            sel = [i for i, r in enumerate(rows) if r['object_category'] == obj and r['action'] == act]
            if not sel:
                print('  %s-%s: no query' % (obj, act)); continue
            print('\n-- %s-%s  (%d queries) --' % (obj, act, len(sel)))
            for si in sel:
                # 用 pair_id 定位该 query 在 npz 里的行
                pid = rows[si]['pair_id']
                if ids is not None:
                    idx = int(np.where(ids == pid)[0][0])
                else:
                    idx = order[pid]
                p = pred[idx].reshape(-1).astype(np.float64)
                g = target[idx].reshape(-1).astype(np.float64) > 0.5
                pos = int(g.sum())
                qs = np.percentile(p, [10, 50, 90, 95, 99])
                print('  query %s: GT pos %d/%d (%.2f%%)  pred mean %.4f  p10 %.3f p50 %.3f p90 %.3f p95 %.3f p99 %.3f max %.3f'
                      % (pid[:8], pos, len(g), 100.0 * pos / len(g), p.mean(),
                         qs[0], qs[1], qs[2], qs[3], qs[4], p.max()))
                # 20 档 IoU 曲线
                curve = []
                for t in np.linspace(0, 1, 20):
                    q = p >= t
                    inter = float((q & g).sum())
                    union = float((q | g).sum())
                    curve.append(inter / max(union, 1.0))
                print('    IoU@t(0..1,20档): ' + ' '.join('%.2f' % c for c in curve))
                print('    aIOU %.4f  best %.4f (t=%.2f)' % (np.mean(curve), max(curve),
                                                              np.linspace(0, 1, 20)[int(np.argmax(curve))]))
                # 正例/负例的预测分布
                if pos and pos < len(g):
                    print('    GT+ mean %.4f  p50 %.3f  p90 %.3f | GT- mean %.4f  p50 %.3f  p90 %.3f'
                          % (p[g].mean(), np.median(p[g]), np.percentile(p[g], 90),
                             p[~g].mean(), np.median(p[~g]), np.percentile(p[~g], 90)))
                    for t in (0.9, 0.95, 0.99):
                        q = p >= t
                        print('      t=%.2f: pred_pos %5d  TP %4d  FP %5d  FN %4d  prec %.3f rec %.3f'
                              % (t, int(q.sum()), int((q & g).sum()), int((q & ~g).sum()),
                                 int((~q & g).sum()),
                                 (q & g).sum() / max(q.sum(), 1),
                                 (q & g).sum() / max(pos, 1)))


if __name__ == '__main__':
    main()
