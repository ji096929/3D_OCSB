"""APGraph 3DIR：拆开"历史激活校准门"对新动作的影响。

三组对照（同一 stage4 权重，不训练）：
  raw       原始推理（bias / tau_vis 来自历史边界迁移，boundary=-20）
  gate_off  所有 expansion 的 bias 置 0、tau_vis 置 1（取消历史标准化边界）
  zero_new  新动作激活全置零（只看 Base 动作是否被抢占）

输出：/data01/jh/afford_diag/apgraph_gate_probe.json
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CODE = '/nas_data/ljy/baseline/3DIR_Order2143_20261109/code'
sys.path.insert(0, CODE)

# data_protocol.runtime 在 import 时读这些环境变量，必须先设好
DATA_ROOT_3DIR = '/nas_data/ljy/all_data/3DIR/release_audit_20261012/extracted/3DIR_release'
os.environ['APGRAPH_DATASET'] = '3dir'
os.environ['APGRAPH_PROTOCOL_ROOT'] = '/nas_data/ljy/baseline/3DIR_Order2143_20261109/protocol'
os.environ['APGRAPH_ASSET_ROOT'] = '/nas_data/ljy/baseline/APGraph_Main/results/3dir/V1'
os.environ['APGRAPH_DATA_ROOT'] = DATA_ROOT_3DIR
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

BASE = '/nas_data/ljy/baseline/APGraph_Main/results/3dir/V1/cluster_no_norm_no_T/base_initial/best.pt'
EXP = ('/nas_data/ljy/baseline/3DIR_Order2143_20261109/APGraph/V2/'
       'cluster_no_norm_no_T/K5_seed1/stage4_expansions.pt')
OUT = '/data01/jh/afford_diag/apgraph_gate_probe.json'

BASE_ACTIONS = ['lift', 'pour', 'sit', 'support', 'ride', 'listen', 'press', 'play', 'carry']


def assess_variant(model, pool, rows, zero_new=False):
    """复刻 develop_incremental.assess，但允许 zero_new 消融。"""
    import torch
    from continual.expansion import tensors
    preds, targets = [], []
    for start in range(0, len(rows), 16):
        e, h, y = tensors(pool, list(range(start, min(start + 16, len(rows)))))
        with torch.no_grad():
            logits, aux = model.decode(e, h, zero_new=zero_new)
        preds.append(logits.sigmoid().cpu().numpy())
        targets.append(y.cpu().numpy())
    return dict(pred=np.concatenate(preds), target=np.concatenate(targets))


def metrics_from_arrays(arrays, rows):
    """按 (object, action) 与按动作聚合官方口径：20 档阈值 aIOU 平均、AUC、AP。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    p, y = arrays['pred'], arrays['target']
    out = []
    for i, r in enumerate(rows):
        pi = p[i].reshape(-1).astype(np.float64)
        gi = y[i].reshape(-1).astype(np.float64) > 0.5
        if gi.sum() == 0:
            continue
        ious = [float((pi >= t)[gi].sum()) / max(float(((pi >= t) | gi).sum()), 1.0)
                for t in np.linspace(0, 1, 20)]
        rec = dict(object=r['object_category'], action=r['action'],
                   aIOU=float(np.mean(ious)), bestIOU=float(np.max(ious)),
                   AUC=float(roc_auc_score(gi, pi)) if (~gi).sum() else float('nan'),
                   AP=float(average_precision_score(gi, pi)),
                   pos=int(gi.sum()))
        out.append(rec)
    return out


def agg(records, key):
    from collections import defaultdict
    d = defaultdict(list)
    for r in records:
        d[r[key]].append(r)
    res = {}
    for k, v in d.items():
        res[k] = dict(aIOU=float(np.mean([x['aIOU'] for x in v])),
                      AUC=float(np.nanmean([x['AUC'] for x in v])),
                      AP=float(np.mean([x['AP'] for x in v])),
                      n=len(v))
    return res


def main():
    import torch
    from infer_graph import load_model
    from continual.expansion import feature_pool
    from fbox.data import read, clear_assets

    torch.set_num_threads(4)
    rows = read('query_all.jsonl')
    print('queries', len(rows), flush=True)
    model = load_model(BASE, EXP, 'cuda')
    pool = feature_pool(model, rows, 'cuda', fixed=True)
    clear_assets()

    variants = {}
    # 1) 原始
    arrays = assess_variant(model, pool, rows)
    variants['raw'] = metrics_from_arrays(arrays, rows)
    print('raw done', flush=True)

    # 2) 取消历史标准化边界
    saved = [(e.bias.clone(), e.tau_vis.clone()) for e in model.expansions]
    for e in model.expansions:
        e.bias.zero_()
        e.tau_vis.fill_(1.0)
    arrays2 = assess_variant(model, pool, rows)
    variants['gate_off'] = metrics_from_arrays(arrays2, rows)
    print('gate_off done', flush=True)

    # 3) 新动作全关
    for i, e in enumerate(model.expansions):
        e.bias.copy_(saved[i][0]); e.tau_vis.copy_(saved[i][1])
    try:
        arrays3 = assess_variant(model, pool, rows, zero_new=True)
        variants['zero_new'] = metrics_from_arrays(arrays3, rows)
        print('zero_new done', flush=True)
    except TypeError as exc:
        print('zero_new unsupported:', exc, flush=True)

    out = {}
    for name, recs in variants.items():
        out[name] = dict(by_action=agg(recs, 'action'), by_object=agg(recs, 'object'),
                         by_pair=recs,
                         mean_aIOU=float(np.mean([r['aIOU'] for r in recs])),
                         mean_AUC=float(np.nanmean([r['AUC'] for r in recs])))
        b = [v['aIOU'] for k, v in out[name]['by_action'].items() if k in BASE_ACTIONS]
        n = [v['aIOU'] for k, v in out[name]['by_action'].items() if k not in BASE_ACTIONS]
        out[name]['base9_aIOU'] = float(np.mean(b)) if b else float('nan')
        out[name]['novel8_aIOU'] = float(np.mean(n)) if n else float('nan')
        print('%-9s base9 %.2f  novel8 %.2f  meanAUC %.2f'
              % (name, 100 * out[name]['base9_aIOU'], 100 * out[name]['novel8_aIOU'],
                 100 * out[name]['mean_AUC']), flush=True)

    json.dump(out, open(OUT, 'w'), indent=1)
    print('->', OUT)

    print('\n=== 逐动作 aIOU：raw / gate_off / zero_new ===')
    acts = sorted(out['raw']['by_action'])
    print('%-12s %8s %9s %9s' % ('action', 'raw', 'gate_off', 'zero_new'))
    for a in acts:
        r = out['raw']['by_action'].get(a, {}).get('aIOU', float('nan'))
        g = out['gate_off']['by_action'].get(a, {}).get('aIOU', float('nan'))
        z = out.get('zero_new', {}).get('by_action', {}).get(a, {}).get('aIOU', float('nan'))
        print('%-12s %8.2f %9.2f %9.2f' % (a, 100 * r, 100 * g, 100 * z))

    print('\n=== 关键配对 ===')
    for obj, act in [('Knife', 'stab'), ('Knife', 'grasp'), ('Baseballbat', 'grasp'),
                     ('Scissors', 'grasp'), ('Bottle', 'wrapgrasp')]:
        print('-- %s-%s' % (obj, act))
        for name in ('raw', 'gate_off', 'zero_new'):
            if name not in out:
                continue
            sel = [r for r in out[name]['by_pair'] if r['object'] == obj and r['action'] == act]
            if sel:
                print('   %-9s aIOU %.3f  bestIOU %.3f  AUC %.2f  AP %.3f  pos %d'
                      % (name, sel[0]['aIOU'], sel[0]['bestIOU'], sel[0]['AUC'], sel[0]['AP'], sel[0]['pos']))


if __name__ == '__main__':
    main()
