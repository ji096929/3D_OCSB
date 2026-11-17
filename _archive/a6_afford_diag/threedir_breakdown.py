"""3DIR / PIAD 误差解剖：逐物体类、逐动作、排序能力 vs 定位精度。

数据来源（全部只读）：
  protocol/query_all.jsonl                      pair_id -> object_category / action / session
  <run>/stage4_query_pairs.json[l]              逐对 aIOU / AUC / SIM / MAE（IAGNet 等有）
  <run>/stage4_query_actions.json               逐动作聚合
  <run>/stage4_query*.npz                       pred/target（APGraph 只有这个，需自行逐对重算）

输出：/data01/jh/afford_diag/threedir_breakdown.json
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROTO = Path('/nas_data/ljy/baseline/3DIR_Order2143_20261109/protocol/query_all.jsonl')

METHODS = {
    'APGraph': {
        'root': '/nas_data/ljy/baseline/3DIR_Order2143_20261109/APGraph/V2/cluster_no_norm_no_T',
        'pat': 'K5_seed%d', 'pairs': 'stage4_query_pairs.json',
        'actions': 'stage4_actions.json', 'npz': 'stage4_query.npz',
    },
    'IAGNet': {
        'root': '/nas_data/ljy/baseline/IAGNet_DINO_HardLabel_ES20/runs/3dir/'
                '3dir_swap_grasp_wrapgrasp_with_lift_pour_v1',
        'pat': 'K5_seed%d', 'pairs': 'evaluations/stage4_query_pairs.jsonl',
        'actions': 'evaluations/stage4_query_actions.json',
        'npz': 'evaluations/stage4_query_predictions.npz',
    },
    'GREAT': {
        'root': '/nas_data/ljy/baseline/GREAT_HardLabel_ES20/runs/3dir/'
                '3dir_swap_grasp_wrapgrasp_with_lift_pour_v1',
        'pat': 'K5_seed%d', 'pairs': 'evaluations/stage4_query_pairs.jsonl',
        'actions': 'stage4_actions.json',
        'npz': 'evaluations/stage4_query_predictions.npz',
    },
    'MIFAG': {
        'root': '/nas_data/ljy/baseline/MIFAG_HardLabel/runs/3dir/'
                '3dir_swap_grasp_wrapgrasp_with_lift_pour_v1',
        'pat': 'K5_seed%d', 'pairs': 'evaluations/stage4_query_pairs.jsonl',
        'actions': 'stage4_actions.json',
        'npz': 'evaluations/stage4_query_predictions.npz',
    },
    'GEAL': {
        'root': '/nas_data/ljy/baseline/GEAL_HardLabel/runs/3dir/'
                '3dir_swap_grasp_wrapgrasp_with_lift_pour_v1',
        'pat': 'K5_seed%d', 'pairs': 'evaluations/stage4_query_pairs.jsonl',
        'actions': 'stage4_actions.json',
        'npz': 'evaluations/stage4_query_predictions.npz',
    },
}

KEYS = ('aIOU', 'AUC', 'AP', 'bestIOU', 'iou05', 'prec', 'rec')
OUT = '/data01/jh/afford_diag/threedir_breakdown.json'


def load_protocol():
    rows = [json.loads(line) for line in PROTO.read_text().splitlines() if line.strip()]
    return {r['pair_id']: r for r in rows}, rows


def read_pairs(path):
    path = Path(path)
    if not path.is_file():
        return None
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def point_metrics(pred, target, thr=0.5):
    """逐对：aIOU@0.5、AUC、AP、最优阈值 aIOU（校准上界）、精确率、召回率。"""
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    g = np.asarray(target, dtype=np.float64).reshape(-1) > 0.5
    if g.sum() == 0:
        return None
    # 官方 3DIR 口径：aIOU 是 20 档阈值（0..1）IoU 的平均，不是 0.5 单点
    ious = [float((p >= t)[g].sum()) / max(float(((p >= t) | g).sum()), 1.0)
            for t in np.linspace(0, 1, 20)]
    aiou = float(np.mean(ious))
    best = float(np.max(ious))
    q = p >= thr
    tp = float((q & g).sum())
    fp = float((q & ~g).sum())
    fn = float((~q & g).sum())
    pos = int(g.sum())
    neg = len(g) - pos
    if neg == 0:
        auc = float('nan')
    else:
        auc = float(roc_auc_score(g, p))
    ap = float(average_precision_score(g, p))
    return dict(aIOU=aiou, AUC=auc, AP=ap, bestIOU=best,
                iou05=tp / max(tp + fp + fn, 1.0),
                prec=tp / max(tp + fp, 1.0), rec=tp / max(tp + fn, 1.0),
                pos=pos, pred=int(q.sum()))


def from_npz(npz_path, meta_by_pid, order_rows):
    """返回 {pair_id: point_metrics}。优先用 npz 里的 pair_ids，否则按 protocol 顺序。"""
    z = np.load(npz_path, allow_pickle=True)
    pred = z['pred']
    target = z['target']
    ids = z['pair_ids'] if 'pair_ids' in z else None
    out = {}
    for i in range(len(pred)):
        if ids is not None:
            pid = str(ids[i])
            if pid not in meta_by_pid:
                continue
        else:
            pid = order_rows[i]['pair_id']
        m = point_metrics(pred[i], target[i])
        if m is not None:
            out[pid] = m
    return out


def aggregate(items):
    out = {}
    for k in KEYS:
        vals = [it[k] for it in items if it is not None and k in it and np.isfinite(it[k])]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['n'] = len([it for it in items if it is not None])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--methods', default='APGraph,IAGNet,GEAL,GREAT,MIFAG')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    meta, rows = load_protocol()
    print('protocol pairs', len(rows))
    result = {'protocol_pairs': len(rows), 'methods': {}}

    for name in args.methods.split(','):
        cfg = METHODS[name]
        per_seed = defaultdict(dict)     # seed -> {pair_id: metrics}
        official = {}
        npz_ok = 0
        for seed in seeds:
            run = Path(cfg['root']) / (cfg['pat'] % seed)
            recs = read_pairs(run / cfg['pairs'])
            got = {}
            if recs:
                for rec in recs:
                    pid = rec.get('pair_id')
                    if pid in meta and np.isfinite(float(rec.get('aIOU', np.nan))):
                        got[pid] = dict(aIOU=float(rec['aIOU']) / 100.0,
                                        AUC=float(rec['AUC']) / 100.0,
                                        AP=float('nan'),
                                        bestIOU=float(rec.get('aIOU', np.nan)) / 100.0,
                                        iou05=float('nan'),
                                        prec=float('nan'), rec=float('nan'))
            nz = run / cfg['npz']
            if nz.is_file():
                npz_ok += 1
                if not got:
                    got = from_npz(nz, meta, rows)
            if got:
                per_seed[seed] = got
            acts = run / cfg['actions']
            if not acts.is_file() and 'evaluations/' in cfg['actions']:
                acts = run / cfg['actions'].split('evaluations/')[-1]
            if acts.is_file():
                official[seed] = json.loads(acts.read_text())

        method = dict(runs_with_pairs=len(per_seed), runs_with_actions=len(official),
                      runs_with_npz=npz_ok)
        # 跨 seed 先平均
        pair_avg = {}
        pids = set().union(*[set(v) for v in per_seed.values()]) if per_seed else set()
        for pid in pids:
            vals = [per_seed[s][pid] for s in per_seed if pid in per_seed[s]]
            pair_avg[pid] = {k: float(np.nanmean([v[k] for v in vals])) for k in KEYS}
        method['pair_coverage'] = len(pair_avg)
        method['mean_auc'] = float(np.nanmean([m['AUC'] for m in pair_avg.values()])) if pair_avg else float('nan')
        method['mean_aiou'] = float(np.nanmean([m['aIOU'] for m in pair_avg.values()])) if pair_avg else float('nan')

        by_obj, by_act, by_oa = defaultdict(list), defaultdict(list), defaultdict(list)
        for pid, m in pair_avg.items():
            by_obj[meta[pid]['object_category']].append(m)
            by_act[meta[pid]['action']].append(m)
            by_oa[(meta[pid]['object_category'], meta[pid]['action'])].append(m)
        method['by_object'] = {o: aggregate(v) for o, v in sorted(by_obj.items())}
        method['by_action'] = {a: aggregate(v) for a, v in sorted(by_act.items())}
        method['by_object_action'] = [dict(object=o, action=a, **aggregate(v))
                                      for (o, a), v in sorted(by_oa.items())]

        gap = sorted([dict(object=meta[p]['object_category'], action=meta[p]['action'],
                           aIOU=m['aIOU'], AUC=m['AUC'], gap=m['AUC'] - m['aIOU'],
                           bestIOU=m['bestIOU'])
                      for p, m in pair_avg.items()], key=lambda x: -x['gap'])
        method['largest_rank_vs_iou_gap'] = gap[:10]
        method['smallest_rank_vs_iou_gap'] = gap[-10:]

        if official:
            act_names = sorted({a for v in official.values() for a in v})
            block = {}
            for a in act_names:
                vals = [v[a] for v in official.values() if a in v]
                block[a] = {k: float(np.mean([x[k] for x in vals]))
                            for k in vals[0] if isinstance(vals[0][k], (int, float))}
            method['official_by_action'] = block
            method['official_mean_aiou'] = float(np.mean([v['aIOU'] for v in block.values()]))
            method['official_mean_auc'] = float(np.mean([v['AUC'] for v in block.values()]))
            # 没有逐对产物时，退回官方逐动作表（GEAL / MIFAG 属于这种情况）
            if not method['by_action']:
                method['by_action'] = {a: dict(aIOU=v['aIOU'] / 100.0, AUC=v['AUC'] / 100.0,
                                               AP=float('nan'), bestIOU=float('nan'),
                                               iou05=float('nan'), prec=float('nan'),
                                               rec=float('nan'), n=len(official))
                                       for a, v in block.items()}
                method['source'] = 'stage actions only (no per-pair dump)'
            else:
                method['source'] = 'per-pair recomputed'

        result['methods'][name] = method
        print('%-7s pairs=%3d actions=%d npz=%d  meanAUC=%.2f  meanAIOU=%.2f  (official aIOU %.2f)'
              % (name, method['pair_coverage'], len(official), npz_ok,
                 method['mean_auc'], 100 * method['mean_aiou'],
                 method.get('official_mean_aiou', float('nan'))), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))

    names = list(result['methods'])
    print('\n=== 逐动作 aIOU（%） ===')
    acts = sorted({a for m in result['methods'].values() for a in m['by_action']})
    print('%-12s' % 'action' + ''.join('%9s' % n for n in names))
    for a in acts:
        print('%-12s' % a + ''.join(
            '%9.2f' % (100 * result['methods'][n]['by_action'].get(a, {}).get('aIOU', np.nan))
            for n in names))

    print('\n=== 逐动作 AUC ===')
    print('%-12s' % 'action' + ''.join('%9s' % n for n in names))
    for a in acts:
        print('%-12s' % a + ''.join(
            '%9.2f' % (100 * result['methods'][n]['by_action'].get(a, {}).get('AUC', np.nan))
            for n in names))

    print('\n=== 逐物体类 aIOU（%） ===')
    objs = sorted({o for m in result['methods'].values() for o in m['by_object']})
    print('%-18s' % 'object' + ''.join('%9s' % n for n in names))
    for o in objs:
        print('%-18s' % o + ''.join(
            '%9.2f' % (100 * result['methods'][n]['by_object'].get(o, {}).get('aIOU', np.nan))
            for n in names))

    print('\n=== APGraph：AUC 高但 aIOU 低（排序好、定位/校准差） ===')
    for r in result['methods']['APGraph']['largest_rank_vs_iou_gap']:
        print('  %-14s %-10s AUC %6.2f  aIOU %5.2f  bestIOU %5.2f  gap %+.3f'
              % (r['object'], r['action'], 100 * r['AUC'], 100 * r['aIOU'],
                 100 * r['bestIOU'], r['gap']))

    print('\n=== APGraph：AUC 与 aIOU 都最低（真的分不开） ===')
    for r in result['methods']['APGraph']['smallest_rank_vs_iou_gap']:
        print('  %-14s %-10s AUC %6.2f  aIOU %5.2f  gap %+.3f'
              % (r['object'], r['action'], 100 * r['AUC'], 100 * r['aIOU'], r['gap']))
    print('\n->', args.out)


if __name__ == '__main__':
    main()
