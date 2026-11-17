"""握持配对在 stage1→stage4 的轨迹：是"学不会"还是"学了又忘"。

- APGraph：stage{0..4}_query.npz（无 pair_ids，按 protocol 顺序）
- IAGNet：evaluations/stage{1..4}_query_predictions.npz + stage{1..4}_query_pairs.jsonl
- GEAL：只有 stage{1..4}_actions.json（逐动作）
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROTO = Path('/nas_data/ljy/baseline/3DIR_Order2143_20261109/protocol/query_all.jsonl')
APG = Path('/nas_data/ljy/baseline/3DIR_Order2143_20261109/APGraph/V2/cluster_no_norm_no_T/K5_seed1')
IAG = Path('/nas_data/ljy/baseline/IAGNet_DINO_HardLabel_ES20/runs/3dir/'
           '3dir_swap_grasp_wrapgrasp_with_lift_pour_v1/K5_seed1')
GEAL = Path('/nas_data/ljy/baseline/GEAL_HardLabel/runs/3dir/'
            '3dir_swap_grasp_wrapgrasp_with_lift_pour_v1/K5_seed1')

TARGETS = [('Baseballbat', 'grasp'), ('Knife', 'grasp'), ('Scissors', 'grasp'),
           ('Bottle', 'wrapgrasp'), ('Mug', 'grasp'), ('Knife', 'stab')]


def aiu(p, g):
    ious = [float((p >= t)[g].sum()) / max(float(((p >= t) | g).sum()), 1.0)
            for t in np.linspace(0, 1, 20)]
    auc = float(roc_auc_score(g, p)) if g.sum() and (~g).sum() else float('nan')
    return float(np.mean(ious)), auc


def load_apgraph(stage):
    z = np.load(APG / ('stage%d_query.npz' % stage), allow_pickle=True)
    return z['pred'], z['target'], None


def load_iagnet(stage):
    npz = IAG / 'evaluations' / ('stage%d_query_predictions.npz' % stage)
    if not npz.is_file():
        return None
    z = np.load(npz, allow_pickle=True)
    return z['pred'], z['target'], z['pair_ids'] if 'pair_ids' in z else None


def main():
    rows = [json.loads(l) for l in PROTO.read_text().splitlines() if l.strip()]
    order = {r['pair_id']: i for i, r in enumerate(rows)}
    idx_of = {(r['object_category'], r['action']): i for i, r in enumerate(rows)}

    print('=== APGraph：stage0→stage4 ===')
    print('%-20s' % 'pair' + ''.join('%14s' % ('s%d aIOU/AUC' % s) for s in range(5)))
    apg_cache = {}
    for s in range(5):
        try:
            apg_cache[s] = load_apgraph(s)
        except FileNotFoundError:
            apg_cache[s] = None
    for obj, act in TARGETS:
        i = idx_of.get((obj, act))
        if i is None:
            continue
        line = '%-20s' % ('%s-%s' % (obj, act))
        for s in range(5):
            if apg_cache[s] is None:
                line += '%14s' % '--'
                continue
            p = apg_cache[s][0][i].reshape(-1).astype(np.float64)
            g = apg_cache[s][1][i].reshape(-1).astype(np.float64) > 0.5
            ai, au = aiu(p, g)
            line += '%14s' % ('%.2f/%.2f' % (100 * ai, 100 * au))
        print(line)

    print('\n=== IAGNet：stage1→stage4 ===')
    print('%-20s' % 'pair' + ''.join('%14s' % ('s%d aIOU/AUC' % s) for s in range(1, 5)))
    for obj, act in TARGETS:
        i = idx_of.get((obj, act))
        if i is None:
            continue
        pid = rows[i]['pair_id']
        line = '%-20s' % ('%s-%s' % (obj, act))
        for s in range(1, 5):
            got = load_iagnet(s)
            if got is None:
                line += '%14s' % '--'
                continue
            pred, targ, ids = got
            k = int(np.where(ids == pid)[0][0]) if ids is not None else order[pid]
            p = pred[k].reshape(-1).astype(np.float64)
            g = targ[k].reshape(-1).astype(np.float64) > 0.5
            ai, au = aiu(p, g)
            line += '%14s' % ('%.2f/%.2f' % (100 * ai, 100 * au))
        print(line)

    print('\n=== GEAL：逐动作 aIOU/AUC 轨迹（没有逐对产物）===')
    acts = ['grasp', 'wrapgrasp', 'stab', 'cut']
    print('%-12s' % 'action' + ''.join('%14s' % ('s%d aIOU/AUC' % s) for s in range(1, 5)))
    for a in acts:
        line = '%-12s' % a
        for s in range(1, 5):
            f = GEAL / ('stage%d_actions.json' % s)
            if not f.is_file():
                line += '%14s' % '--'
                continue
            d = json.loads(f.read_text())
            if a in d:
                line += '%14s' % ('%.2f/%.2f' % (d[a]['aIOU'], d[a]['AUC']))
            else:
                line += '%14s' % '--'
        print(line)

    print('\n=== IAGNet 逐动作（对照 GEAL）===')
    for a in acts:
        line = '%-12s' % a
        for s in range(1, 5):
            f = IAG / 'evaluations' / ('stage%d_query_actions.json' % s)
            if not f.is_file():
                line += '%14s' % '--'
                continue
            d = json.loads(f.read_text())
            line += '%14s' % ('%.2f/%.2f' % (d[a]['aIOU'], d[a]['AUC'])) if a in d else '%14s' % '--'
        print(line)


if __name__ == '__main__':
    main()
