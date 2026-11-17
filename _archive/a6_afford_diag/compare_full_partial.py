"""full vs partial 的逐类/逐动作对比（用 TP/FP/FN 算 IoU 与召回）。"""
import json
from collections import defaultdict

FULL = '/data01/jh/afford_diag/full_k5/per_pair_errors.json'
PART = '/data01/jh/afford_diag/partial_k5/per_pair_errors.json'
OUT = '/data01/jh/afford_diag/compare_full_partial.json'


def load(p):
    d = defaultdict(lambda: dict(tp=0, fp=0, fn=0, pos=0))
    for r in json.load(open(p)):
        k = (r['cls'], r['aff'])
        for kk in ('tp', 'fp', 'fn', 'pos'):
            d[k][kk] += r[kk]
    return d


def metrics(v):
    return (v['tp'] / max(v['tp'] + v['fp'] + v['fn'], 1),
            v['tp'] / max(v['tp'] + v['fn'], 1),
            v['tp'] / max(v['tp'] + v['fp'], 1))


def main():
    f = load(FULL)
    p = load(PART)
    keys = sorted(set(f) & set(p))

    print('=== 逐 (class, aff)：full vs partial ===')
    print('%-18s %-12s %7s %7s %7s %7s %7s' %
          ('class', 'aff', 'IoU_f', 'IoU_p', 'dIoU', 'Rec_f', 'Rec_p'))
    rows = []
    for k in keys:
        iou_f, rec_f, pre_f = metrics(f[k])
        iou_p, rec_p, pre_p = metrics(p[k])
        rows.append(dict(cls=k[0], aff=k[1], iou_full=iou_f, iou_part=iou_p,
                         d=iou_p - iou_f, rec_full=rec_f, rec_part=rec_p,
                         pre_full=pre_f, pre_part=pre_p))
    for r in sorted(rows, key=lambda r: r['d'])[:12]:
        print('%-18s %-12s %7.3f %7.3f %+7.3f %7.3f %7.3f'
              % (r['cls'], r['aff'], r['iou_full'], r['iou_part'], r['d'],
                 r['rec_full'], r['rec_part']))

    # 按类聚合
    cls_agg = defaultdict(lambda: dict(tp=0, fp=0, fn=0))
    for (c, a), v in f.items():
        for kk in ('tp', 'fp', 'fn'):
            cls_agg[c][kk + '_f'] = cls_agg[c].get(kk + '_f', 0) + v[kk]
    for (c, a), v in p.items():
        for kk in ('tp', 'fp', 'fn'):
            cls_agg[c][kk + '_p'] = cls_agg[c].get(kk + '_p', 0) + v[kk]

    print('\n=== 逐物体类：IoU 掉分排序 ===')
    print('%-20s %8s %8s %8s' % ('class', 'IoU_full', 'IoU_part', 'dIoU'))
    cls_rows = []
    for c, v in cls_agg.items():
        iou_f = v['tp_f'] / max(v['tp_f'] + v['fp_f'] + v['fn_f'], 1)
        iou_p = v['tp_p'] / max(v['tp_p'] + v['fp_p'] + v['fn_p'], 1)
        cls_rows.append((c, iou_f, iou_p, iou_p - iou_f))
    for c, a, b, d in sorted(cls_rows, key=lambda t: t[3]):
        print('%-20s %8.3f %8.3f %+8.3f' % (c, a, b, d))

    # 按动作聚合
    aff_agg = defaultdict(lambda: dict())
    for (c, a), v in f.items():
        for kk in ('tp', 'fp', 'fn'):
            aff_agg[a][kk + '_f'] = aff_agg[a].get(kk + '_f', 0) + v[kk]
    for (c, a), v in p.items():
        for kk in ('tp', 'fp', 'fn'):
            aff_agg[a][kk + '_p'] = aff_agg[a].get(kk + '_p', 0) + v[kk]

    print('\n=== 逐动作：IoU 掉分排序 ===')
    print('%-16s %8s %8s %8s' % ('affordance', 'IoU_full', 'IoU_part', 'dIoU'))
    aff_rows = []
    for a, v in aff_agg.items():
        iou_f = v['tp_f'] / max(v['tp_f'] + v['fp_f'] + v['fn_f'], 1)
        iou_p = v['tp_p'] / max(v['tp_p'] + v['fp_p'] + v['fn_p'], 1)
        aff_rows.append((a, iou_f, iou_p, iou_p - iou_f))
    for a, x, y, d in sorted(aff_rows, key=lambda t: t[3]):
        print('%-16s %8.3f %8.3f %+8.3f' % (a, x, y, d))

    json.dump(dict(pairs=rows,
                   by_class=[dict(cls=c, iou_full=a, iou_part=b, d=d)
                             for c, a, b, d in cls_rows],
                   by_aff=[dict(aff=a, iou_full=x, iou_part=y, d=d)
                           for a, x, y, d in aff_rows]),
              open(OUT, 'w'), indent=1)
    print('\n->', OUT)


if __name__ == '__main__':
    main()
