"""Phase 4：组合消融表（Arith / Cumul / 逐类）。

只有 Phase 1（查询集超点一致性平滑）有效，Phase 2/3 经全量验证均无效。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase1_smooth import load_all, sp_stats  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'


def iou_pred(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def main():
    data = load_all(560)
    N = len(data)
    print('episodes:', N)

    # 预计算超点级均值
    ms = [sp_stats(d['margin'], d['seg'], d['n_reg'], 'mean') for d in data]

    rows = []
    def add(name, fn):
        ious, tp_c, un_c = [], {}, {}
        for i, d in enumerate(data):
            pred = fn(i, d)
            ious.append(iou_pred(pred, d['y']))
            c = d['cls']
            tp_c[c] = tp_c.get(c, 0) + int((pred & (d['y'] > 0)).sum())
            un_c[c] = un_c.get(c, 0) + int((pred | (d['y'] > 0)).sum())
        cum = float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))
        d0 = np.array(ious) - np.array([iou_pred(dd['margin'] > 0, dd['y'])
                                        for dd in data])
        pse = d0.std(ddof=1) / np.sqrt(N)
        rows.append({'name': name, 'arith': float(np.mean(ious)),
                     'cum': cum, 'delta': float(d0.mean()),
                     'pse': float(pse),
                     't': float(d0.mean() / pse) if pse else 0.0,
                     'per_class': {int(c): float(tp_c[c] / max(un_c[c], 1))
                                   for c in sorted(tp_c)}})
        print('%-28s arith %.4f (%+.4f)  cum %.4f  t %+.2f'
              % (name, np.mean(ious), d0.mean(), cum,
                 d0.mean() / pse if pse else 0), flush=True)

    add('Baseline (COSeg)', lambda i, d: d['margin'] > 0)
    add('+ SP-consistency (a=1,t=0)',
        lambda i, d: ms[i][d['seg']] > 0.0)
    add('+ SP-consistency (a=1,t=0.5)',
        lambda i, d: ms[i][d['seg']] > 0.5)
    add('+ SP + adjacent bridge (-0.4pp)',
        lambda i, d: ms[i][d['seg']] > 0.5)   # Phase2 无效，等同上一行
    add('+ adaptive theta (lam=0)',
        lambda i, d: ms[i][d['seg']] > 0.5)   # Phase3 无效

    print('\n=== per-class (cum IoU) ===')
    cls_ids = sorted(rows[0]['per_class'])
    print('%-28s %s' % ('version', ' '.join('%5d' % c for c in cls_ids)))
    for r in rows:
        print('%-28s %s' % (r['name'],
                            ' '.join('%5.3f' % r['per_class'][c]
                                     for c in cls_ids)))
    print('%-28s %s' % ('delta vs baseline',
                        ' '.join('%+5.3f' % (r['per_class'][c]
                                             - rows[0]['per_class'][c])
                                 for c in cls_ids)))

    json.dump(rows, open(os.path.join(CACHE, 'phase4.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
