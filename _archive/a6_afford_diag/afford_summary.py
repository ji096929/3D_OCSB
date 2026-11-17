"""汇总：Base/Novel 拆分 + 逐类/逐动作 mAP + 误差性质。"""
import csv
import json
import os
from collections import defaultdict

PAIR = '/nas_data/ljy/AffordanceNet-main/docs/fixed_graph_main_results_20261005/per_pair.csv'
MANI = '/nas_data/ljy/AffordanceNet-main/archive/coverage_protocol_20261004/manifests/k5_seed1.json'
ERR = '/data01/jh/afford_diag/full_k5/per_pair_errors.json'
OUT = '/data01/jh/afford_diag'


def main():
    man = json.load(open(MANI))
    base_cls = set(man['sessions'][0]['new_classes'])
    novel_cls = set()
    for s in man['sessions'][1:]:
        novel_cls |= set(s['new_classes'])

    rows = []
    with open(PAIR, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append(dict(bb=r['backbone'], shot=int(r['shot']), seed=int(r['seed']),
                             cls=r['object_class'], aff=r['affordance'],
                             AP=float(r['AP']), aIOU=float(r['aIOU']),
                             AUC=float(r['AUC']), n=int(r['instances'])))

    print('=== 1) Base/Novel 分组 mAP（dgcnn，5 seeds 平均）===')
    print('%-6s %8s %8s %8s %8s %8s' % ('shot', 'Base', 'Novel', 'All', 'BaseAUC', 'NovelAUC'))
    for sh in (1, 2, 5):
        sub = [r for r in rows if r['bb'] == 'dgcnn' and r['shot'] == sh]
        b = [r['AP'] for r in sub if r['cls'] in base_cls]
        nv = [r['AP'] for r in sub if r['cls'] in novel_cls]
        ba = [r['AUC'] for r in sub if r['cls'] in base_cls]
        na = [r['AUC'] for r in sub if r['cls'] in novel_cls]
        print('%-6d %8.2f %8.2f %8.2f %8.2f %8.2f'
              % (sh, sum(b) / len(b), sum(nv) / len(nv),
                 sum(r['AP'] for r in sub) / len(sub),
                 sum(ba) / len(ba), sum(na) / len(na)))

    print('\n=== 2) 12 个 Novel 类逐类 mAP（dgcnn，per-pair 聚合，5 seeds）===')
    print('%-20s %8s %8s %8s %8s' % ('class', 'k1', 'k2', 'k5', 'k5-k1'))
    for c in sorted(novel_cls):
        vals = {}
        for sh in (1, 2, 5):
            sub = [r['AP'] for r in rows
                   if r['bb'] == 'dgcnn' and r['shot'] == sh and r['cls'] == c]
            vals[sh] = sum(sub) / len(sub) if sub else 0
        print('%-20s %8.2f %8.2f %8.2f %+8.2f'
              % (c, vals[1], vals[2], vals[5], vals[5] - vals[1]))

    print('\n=== 3) 18 个动作逐动作 mAP（dgcnn，5 seeds）===')
    print('%-16s %8s %8s %8s %8s' % ('affordance', 'k1', 'k2', 'k5', 'k5-k1'))
    affs = man['affordances']
    for a in affs:
        vals = {}
        for sh in (1, 2, 5):
            sub = [r['AP'] for r in rows
                   if r['bb'] == 'dgcnn' and r['shot'] == sh and r['aff'] == a]
            vals[sh] = sum(sub) / len(sub) if sub else 0
        print('%-16s %8.2f %8.2f %8.2f %+8.2f'
              % (a, vals[1], vals[2], vals[5], vals[5] - vals[1]))

    print('\n=== 4) 误差性质（5-shot seed1，逐点 0.5 阈值）===')
    err = json.load(open(ERR))
    grouped = defaultdict(lambda: dict(tp=0, fp=0, fn=0, pos=0))
    for r in err:
        g = grouped[r['split']]
        for k in ('tp', 'fp', 'fn', 'pos'):
            g[k] += r[k]
    print('%-8s %7s %9s %9s %9s %9s %9s' %
          ('split', 'pairs', 'TP', 'FP', 'FN', '召回', 'FP/pos'))
    for sp in ('novel', 'base'):
        g = grouped[sp]
        npair = len([r for r in err if r['split'] == sp])
        print('%-8s %7d %9d %9d %9d %9.3f %9.3f'
              % (sp, npair, g['tp'], g['fp'], g['fn'],
                 g['tp'] / max(g['tp'] + g['fn'], 1), g['fp'] / max(g['pos'], 1)))

    print('\n=== 5) 有效配对内：召回最低 / 溢出最高的 (class, aff) ===')
    eff = sorted(err, key=lambda r: r['recall'])[:8]
    print('-- 召回最低 --')
    for r in eff:
        print('%-18s %-12s %-6s recall=%.3f prec=%.3f fp=%d fn=%d'
              % (r['cls'], r['aff'], r['split'], r['recall'], r['precision'],
                 r['fp'], r['fn']))
    print('-- 溢出最高 --')
    for r in sorted(err, key=lambda r: -r['overflow'])[:8]:
        print('%-18s %-12s %-6s overflow=%.2f prec=%.3f fp=%d pos=%d'
              % (r['cls'], r['aff'], r['split'], r['overflow'], r['precision'],
                 r['fp'], r['pos']))

    json.dump(dict(base_classes=sorted(base_cls), novel_classes=sorted(novel_cls)),
              open(os.path.join(OUT, 'split_definition.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
