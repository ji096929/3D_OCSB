"""APGraph 3DIR：输出尺度校准 + Base 响应扩散检验。

三组对照（同一 stage4 权重，不训练）：
  raw       原始 logits
  zero_old  Base 127 个属性的视觉门全部置 0，只留 expansion
  zero_new  expansion 的视觉门置 0，只留 Base

温度校准（严格全局常数，不在 Novel 上挑参）：
  p = sigmoid((logit - c) / T)
  在 Base 9 动作上网格选 (c, T)，再原样应用到 Novel 8。

输出：/data01/jh/afford_diag/apgraph_scale_probe.json
"""
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE = '/nas_data/ljy/baseline/3DIR_Order2143_20261109/code'
sys.path.insert(0, CODE)

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
OUT = '/data01/jh/afford_diag/apgraph_scale_probe.json'

BASE_ACTIONS = ['lift', 'pour', 'sit', 'support', 'ride', 'listen', 'press', 'play', 'carry']
NOVEL_ACTIONS = ['grasp', 'wrapgrasp', 'lay', 'move', 'cut', 'stab', 'open', 'pull']


def decode_variant(model, e, h, zero_new=False, zero_old=False):
    """复刻 ContinualGrounder.decode，额外支持 zero_old。"""
    import torch
    import torch.nn.functional as F
    v, a, g = model.bank()
    base_logits = e @ v[:model.m_base].T / model.tau_v + model.bias
    base_alpha = 0.5 * (1 + torch.erf(base_logits / math.sqrt(2)))
    pieces = [base_alpha] + [exp.visual_activation(e) for exp in model.expansions]
    alpha = torch.cat(pieces, -1)
    if zero_old:
        alpha = alpha.clone()
        alpha[:, :model.m_base] = 0
    if zero_new:
        alpha = alpha.clone()
        alpha[:, model.m_base:] = 0
    response = h @ F.normalize(model.raw_attributes(), dim=-1).T
    node = response * alpha[:, None, :]
    z = node @ a
    return model.grounding_head(z)[:, :, 0]           # [B, N]


def collect(model, pool, rows, **kwargs):
    """返回 [(logit[np.ndarray], gt[np.ndarray], action, object)]。"""
    import torch
    from continual.expansion import tensors
    out = []
    for start in range(0, len(rows), 16):
        ids = list(range(start, min(start + 16, len(rows))))
        e, h, y = tensors(pool, ids)
        with torch.no_grad():
            logits = decode_variant(model, e, h, **kwargs)
        lg = logits.float().cpu().numpy().reshape(len(ids), -1)
        gt = (y.float().cpu().numpy().reshape(len(ids), -1) > 0.5)
        for k, i in enumerate(ids):
            out.append((lg[k], gt[k], rows[i]['action'], rows[i]['object_category']))
    return out


def aiu(logit, gt, c=0.0, T=1.0):
    p = 1.0 / (1.0 + np.exp(-(logit - c) / T))
    ious = [float((p >= t)[gt].sum()) / max(float(((p >= t) | gt).sum()), 1.0)
            for t in np.linspace(0, 1, 20)]
    return float(np.mean(ious))


def agg(items, actions):
    vals = [aiu(lg, gt) for lg, gt, act, obj in items if act in actions]
    return float(np.mean(vals)), len(vals)


def calib_agg(items, actions, c, T):
    vals = [aiu(lg, gt, c, T) for lg, gt, act, obj in items if act in actions]
    return float(np.mean(vals)), len(vals)


def main():
    import torch
    from infer_graph import load_model
    from continual.expansion import feature_pool
    from fbox.data import read, clear_assets

    torch.set_num_threads(4)
    rows = read('query_all.jsonl')
    model = load_model(BASE, EXP, 'cuda')
    pool = feature_pool(model, rows, 'cuda', fixed=True)
    clear_assets()

    variants = {}
    for name, kw in (('raw', {}), ('zero_old', dict(zero_old=True)),
                     ('zero_new', dict(zero_new=True))):
        items = collect(model, pool, rows, **kw)
        variants[name] = items
        b, nb = agg(items, BASE_ACTIONS)
        n, nn_ = agg(items, NOVEL_ACTIONS)
        print('%-9s Base9 %.2f (n=%d)  Novel8 %.2f (n=%d)'
              % (name, 100 * b, nb, 100 * n, nn_), flush=True)

    # 温度校准：只在 Base 9 上选 (c, T)
    best = None
    for T in (1, 2, 3, 5, 8, 12, 20, 30):
        for c in np.linspace(-30, 30, 13):
            v, _ = calib_agg(variants['raw'], BASE_ACTIONS, float(c), float(T))
            if best is None or v > best[0]:
                best = (v, float(c), float(T))
    base_cal, c_best, T_best = best
    novel_cal, _ = calib_agg(variants['raw'], NOVEL_ACTIONS, c_best, T_best)
    print('\n在 Base9 上选到 c=%.1f  T=%.1f（Base9 %.2f）' % (c_best, T_best, 100 * base_cal))
    print('原样应用到 Novel8: %.2f' % (100 * novel_cal))

    # 完整的 Novel8 对 T/c 的敏感性（不挑参，只展示）
    print('\nNovel8 aIOU 随 (c, T) 的变化（行=c，列=T）')
    Ts = [1, 2, 3, 5, 8, 12, 20, 30]
    print('%-7s' % 'c\\T' + ''.join('%8d' % t for t in Ts))
    grid = {}
    for c in np.linspace(-30, 30, 13):
        line = '%-7.1f' % c
        for T in Ts:
            v, _ = calib_agg(variants['raw'], NOVEL_ACTIONS, float(c), float(T))
            grid['%.1f_%d' % (c, T)] = 100 * v
            line += '%8.2f' % (100 * v)
        print(line)

    res = dict(
        raw=dict(base9=100 * agg(variants['raw'], BASE_ACTIONS)[0],
                 novel8=100 * agg(variants['raw'], NOVEL_ACTIONS)[0]),
        zero_old=dict(base9=100 * agg(variants['zero_old'], BASE_ACTIONS)[0],
                      novel8=100 * agg(variants['zero_old'], NOVEL_ACTIONS)[0]),
        zero_new=dict(base9=100 * agg(variants['zero_new'], BASE_ACTIONS)[0],
                      novel8=100 * agg(variants['zero_new'], NOVEL_ACTIONS)[0]),
        calib=dict(c=c_best, T=T_best, base9=100 * base_cal, novel8=100 * novel_cal),
        grid=grid,
        per_pair={
            'raw': [dict(action=a, obj=o, aIOU=100 * aiu(lg, gt))
                    for lg, gt, a, o in variants['raw']],
            'zero_old': [dict(action=a, obj=o, aIOU=100 * aiu(lg, gt))
                         for lg, gt, a, o in variants['zero_old']],
        })
    json.dump(res, open(OUT, 'w'), indent=1)
    print('\n->', OUT)

    # 逐动作：zero_old 的影响
    print('\n=== 逐动作 aIOU：raw / zero_old ===')
    da, db = defaultdict(list), defaultdict(list)
    for lg, gt, a, o in variants['raw']:
        da[a].append(aiu(lg, gt))
    for lg, gt, a, o in variants['zero_old']:
        db[a].append(aiu(lg, gt))
    print('%-11s %9s %9s %8s' % ('action', 'raw', 'zero_old', 'd'))
    for a in sorted(da):
        x, y = 100 * np.mean(da[a]), 100 * np.mean(db[a])
        print('%-11s %9.2f %9.2f %+8.2f' % (a, x, y, y - x))


if __name__ == '__main__':
    main()
