"""从 ljy 已有的 per_pair.csv 做逐物体类 / 逐动作拆解（只读，不跑模型）。

per_pair.csv 列：backbone, shot, seed, object_class, affordance,
                  AP, AUC, F1, aIOU, MSE, instances
每个 (backbone, shot, seed, object_class, affordance) 一行。
输出写到 /data01/jh/afford_diag/。
"""
import csv
import json
import os
from collections import defaultdict

SRC = '/nas_data/ljy/AffordanceNet-main/docs/fixed_graph_main_results_20261005/per_pair.csv'
OUT = '/data01/jh/afford_diag'


def load():
    rows = []
    with open(SRC, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append({
                'backbone': r['backbone'],
                'shot': int(r['shot']),
                'seed': int(r['seed']),
                'cls': r['object_class'],
                'aff': r['affordance'],
                'AP': float(r['AP']),
                'AUC': float(r['AUC']),
                'F1': float(r['F1']),
                'aIOU': float(r['aIOU']),
                'MSE': float(r['MSE']),
                'n': int(r['instances']),
            })
    return rows


def agg(rows, key):
    """按 key 聚合 AP（等权，即 per-pair 的 mAP）。"""
    d = defaultdict(list)
    cnt = defaultdict(int)
    for r in rows:
        d[r[key]].append(r['AP'])
        if key == 'cls':
            cnt[r[key]] += r['n']
    out = []
    for k, v in d.items():
        out.append((k, sum(v) / len(v), len(v), cnt.get(k, 0)))
    return sorted(out, key=lambda t: t[1])


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    print('rows:', len(rows))
    bbs = sorted(set(r['backbone'] for r in rows))
    shots = sorted(set(r['shot'] for r in rows))
    print('backbones:', bbs, 'shots:', shots)
    print('object classes:', len(set(r['cls'] for r in rows)),
          'affordances:', len(set(r['aff'] for r in rows)))

    result = {}
    for bb in bbs:
        for sh in shots:
            sub = [r for r in rows if r['backbone'] == bb and r['shot'] == sh]
            by_cls = agg(sub, 'cls')
            by_aff = agg(sub, 'aff')
            key = '%s_k%d' % (bb, sh)
            result[key] = {
                'n_pairs': len(sub),
                'mAP_all': sum(r['AP'] for r in sub) / len(sub),
                'by_class': [{'cls': k, 'mAP': v, 'n_pairs': c, 'instances': n}
                             for k, v, c, n in by_cls],
                'by_affordance': [{'aff': k, 'mAP': v, 'n_pairs': c}
                                  for k, v, c, _ in by_aff],
            }
            print('\n===== %s =====' % key)
            print('mAP_all = %.2f   pairs = %d' % (result[key]['mAP_all'], len(sub)))
            print('\n-- 逐物体类 mAP（低->高）--')
            print('%-22s %8s %7s %8s' % ('class', 'mAP', 'pairs', 'inst'))
            for k, v, c, n in by_cls:
                print('%-22s %8.2f %7d %8d' % (k, v, c, n))
            print('\n-- 逐动作 mAP（低->高）--')
            print('%-16s %8s %7s' % ('affordance', 'mAP', 'pairs'))
            for k, v, c, _ in by_aff:
                print('%-16s %8.2f %7d' % (k, v, c))

    with open(os.path.join(OUT, 'perpair_breakdown.json'), 'w') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print('\n->', os.path.join(OUT, 'perpair_breakdown.json'))


if __name__ == '__main__':
    main()
