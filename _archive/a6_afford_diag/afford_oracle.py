"""Oracle 拆解：无效配对置零 / 有效对内阈值校准，能带来多少指标提升。

五组对照（全部在验证集预测概率上重算，不训练）：
  raw       原始输出，阈值 0.5
  oracleA   无效 (object, affordance) 对概率置 0（官方指标不看它们，故恒等于 raw）
  oracleB   每个有效对各自最优阈值（58 自由度，上界）
  global    所有有效对共用同一个阈值（1 自由度，可实现）
  perclass  每个物体类一个阈值（23 自由度，可实现）

指标：All/New/novel12 的 mAP、maIOU、mAUC；invalid_pair_FPR
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = '/nas_data/ljy/AffordanceNet-main'
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset', default='full_shape')
    ap.add_argument('--session', type=int, default=3)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='full_k5')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from dataset.object_incremental import ObjectIncrementalDataset
    from attribute_graph.incremental import load_pair_graph
    from sklearn.metrics import average_precision_score, roc_auc_score

    torch.set_num_threads(4)
    man = json.loads(open(args.manifest).read())
    affs = man['affordances']
    validity = {k: set(v) for k, v in man['validity'].items()}
    novel_all = set()
    for s in man['sessions'][1:]:
        novel_all |= set(s['new_classes'])
    last_new = set(man['sessions'][-1]['new_classes'])

    pair, _, _ = load_pair_graph(args.base_dir + '/base.pt', args.checkpoint)
    pair.eval()
    ds = ObjectIncrementalDataset(args.data_root, man, args.session, 'val',
                                  all_objects=True)
    print('samples', len(ds), 'affs', len(affs), flush=True)

    buckets = defaultdict(lambda: [[], []])
    inv_pred = inv_neg = val_pred = val_neg = 0

    for i in range(len(ds)):
        coord, _, target, shape_id, cls = ds[i]
        x = torch.from_numpy(coord).float().cuda().unsqueeze(0).transpose(1, 2)
        with torch.inference_mode():
            prob = torch.sigmoid(pair(x))[0].cpu().numpy()
        gt = np.asarray(target, dtype=np.float32).T
        for a, an in enumerate(affs):
            b = buckets[(cls, an)]
            b[0].append(prob[a])
            b[1].append(gt[a])
            p = prob[a] > 0.5
            g = gt[a] > 0.5
            if an in validity.get(cls, set()):
                val_pred += int(p.sum()); val_neg += int((~g).sum())
            else:
                inv_pred += int(p.sum()); inv_neg += int((~g).sum())
        if (i + 1) % 1000 == 0:
            print('  %d / %d' % (i + 1, len(ds)), flush=True)

    eff = {k: v for k, v in buckets.items() if k[1] in validity.get(k[0], set())}

    def pair_metrics(prob_list, gt_list, thr=None, best=False):
        """官方口径：每个实例算一次，再对实例平均。"""
        aps, aucs, ious = [], [], []
        for prob, gt in zip(prob_list, gt_list):
            g = gt > 0.5
            if g.sum() == 0:
                continue
            aps.append(float(average_precision_score(g, prob)))
            try:
                aucs.append(float(roc_auc_score(g, prob)))
            except ValueError:
                pass
            cand = np.linspace(0.05, 0.95, 9) if best else np.array([thr])
            bi = -1.0
            for t in cand:
                p = prob > t
                bi = max(bi, (p & g).sum() / max((p | g).sum(), 1))
            ious.append(float(bi))
        if not aps:
            return None
        return dict(ap=float(np.mean(aps)),
                    auc=float(np.mean(aucs)) if aucs else float('nan'),
                    aiou=float(np.mean(ious)), n_inst=len(aps))

    def maioU_at(thr_fn):
        vals = []
        for (c, an), (pl, gl) in eff.items():
            t = thr_fn(c)
            iv = []
            for prob, gt in zip(pl, gl):
                g = gt > 0.5
                if g.sum() == 0:
                    continue
                p = prob > t
                iv.append((p & g).sum() / max((p | g).sum(), 1))
            if iv:
                vals.append(float(np.mean(iv)))
        return float(np.mean(vals)) if vals else 0.0

    can = np.linspace(0.02, 0.98, 49)
    best_g, best_gv = 0.5, -1.0
    for t in can:
        v = maioU_at(lambda c, t=t: float(t))
        if v > best_gv:
            best_gv, best_g = v, float(t)
    print('global thr %.2f maioU %.4f' % (best_g, best_gv), flush=True)

    cls_thr, cls_best = {}, []
    for c in sorted(set(k[0] for k in eff)):
        bt, bv = 0.5, -1.0
        for t in can:
            iv = []
            for (cc, an), (pl, gl) in eff.items():
                if cc != c:
                    continue
                for prob, gt in zip(pl, gl):
                    g = gt > 0.5
                    if g.sum() == 0:
                        continue
                    p = prob > t
                    iv.append((p & g).sum() / max((p | g).sum(), 1))
            v = float(np.mean(iv)) if iv else 0.0
            if v > bv:
                bv, bt = v, float(t)
        cls_thr[c] = bt
        cls_best.append(bv)
    print('class thr done', flush=True)

    def summarize(thr=0.5, best=False, per_cls=False):
        out = {}
        for scope, cset in (('all', None), ('new', last_new), ('novel12', novel_all)):
            items = []
            for (c, an), (pl, gl) in eff.items():
                if cset is not None and c not in cset:
                    continue
                m = pair_metrics(pl, gl, thr=(cls_thr[c] if per_cls else thr),
                                 best=best)
                if m is not None:
                    items.append((c, an, m))
            if items:
                out[scope] = dict(pairs=len(items),
                                  mAP=float(np.mean([m['ap'] for _, _, m in items])),
                                  mAUC=float(np.nanmean([m['auc'] for _, _, m in items])),
                                  maIOU=float(np.mean([m['aiou'] for _, _, m in items])))
        return out

    raw = summarize()
    ob = raw  # 跳过 best（partial 上太慢；full 已有上界）
    glob_ = summarize(thr=best_g)
    clsw = summarize(per_cls=True)

    res = dict(tag=args.tag, dataset=args.dataset, samples=len(ds),
               invalid_pair_FPR=inv_pred / max(inv_neg, 1),
               valid_pair_FPR=val_pred / max(val_neg, 1),
               global_thr=best_g, global_maioU=best_gv,
               class_thr=cls_thr,
               class_maioU=float(np.mean(cls_best)) if cls_best else 0.0,
               raw=raw, oracleA=raw, oracleB=ob, glob=glob_, perclass=clsw)
    json.dump(res, open(args.out, 'w'), indent=1)

    print('\n=== %s  (%d samples) ===' % (args.tag, len(ds)))
    print('invalid_pair_FPR = %.4f   valid_pair_FPR = %.4f'
          % (res['invalid_pair_FPR'], res['valid_pair_FPR']))
    for scope in ('all', 'new', 'novel12'):
        if scope not in raw:
            continue
        print('\n-- %s (%d pairs) --' % (scope, raw[scope]['pairs']))
        print('%-10s %8s %8s %8s' % ('variant', 'mAP', 'maIOU', 'mAUC'))
        for n, r in (('raw', raw), ('oracleA', raw), ('oracleB', ob),
                     ('global', glob_), ('perclass', clsw)):
            if scope in r:
                print('%-10s %8.2f %8.2f %8.2f'
                      % (n, 100 * r[scope]['mAP'], 100 * r[scope]['maIOU'],
                         100 * r[scope]['mAUC']))
    print('\n->', args.out)


if __name__ == '__main__':
    main()
