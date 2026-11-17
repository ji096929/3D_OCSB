"""Affordance 基线四维错题诊断（只读 NAS，输出到 /data01/jh/afford_diag/）。

复用 ljy 的 PairGraph + ObjectIncrementalDataset，不修改其任何文件。

输出：
  per_pair_errors.json    每个 (object_class, affordance) 的 TP/FP/FN 与召回/溢出
  invalid_pair_errors.json 无效物体-动作组合的误报
  action_confusion.json   新类物体上 GT 动作 vs 预测动作的混淆
  worst_samples.npz       最差 5 个样本的 coord / gt / pred（供 ply 导出）

用法：
  PYTHONDONTWRITEBYTECODE=1 python afford_error_diag.py \
      --base-dir .../coverage_main_20261004/dgcnn/base \
      --checkpoint .../fixed_graph_ablation_20261005/runs/full_seed1/session_3/model.pt \
      --dataset full_shape --session 3 --gpu 0 --out /data01/jh/afford_diag
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = '/nas_data/ljy/AffordanceNet-main'
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset', default='full_shape')
    ap.add_argument('--session', type=int, default=3)
    ap.add_argument('--manifest', default=os.path.join(
        ROOT, 'archive/coverage_protocol_20261004/manifests/k5_seed1.json'))
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out', default='/data01/jh/afford_diag')
    ap.add_argument('--topk-worst', type=int, default=5)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from dataset.object_incremental import ObjectIncrementalDataset
    from attribute_graph.incremental import load_pair_graph

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)

    manifest = json.loads(open(args.manifest).read())
    affs = manifest['affordances']
    validity = {k: set(v) for k, v in manifest['validity'].items()}
    novel = set()
    for s in manifest['sessions'][1:]:
        novel |= set(s['new_classes'])
    base = set(manifest['sessions'][0]['new_classes'])

    data_root = args.data_root or '/nas_data/ljy/all_data/3D_AffordanceNet/%s' % args.dataset
    pair, states, audit = load_pair_graph(args.base_dir + '/base.pt', args.checkpoint)
    pair.eval()
    ds = ObjectIncrementalDataset(data_root, manifest, args.session, 'val',
                                  all_objects=True)
    print('samples', len(ds), 'affs', len(affs),
          'novel', len(novel), 'base', len(base), flush=True)

    # (cls, aff) -> counts
    stats = {}
    invalid = {}          # (cls, aff) -> pred positive points
    conf = {}             # (gt_aff, pred_aff) -> points, 只在 novel 类
    per_sample = []       # (iou, idx, cls, shape_id, valid_affs)

    for i in range(len(ds)):
        coord, _, target, shape_id, cls = ds[i]
        x = torch.from_numpy(coord).float().cuda().unsqueeze(0).transpose(1, 2)
        with torch.inference_mode():
            logits = pair(x)                       # [1, 18, N]
            prob = torch.sigmoid(logits)[0]        # [18, N]
        prob = prob.cpu().numpy()
        gt = np.asarray(target, dtype=np.float32).T  # [18, N]
        pred = prob > 0.5
        gtm = gt > 0.5
        valid_affs = validity.get(cls, set())

        ious = []
        for a, an in enumerate(affs):
            p = pred[a]
            g = gtm[a]
            if an in valid_affs:
                tp = int((p & g).sum())
                fp = int((p & ~g).sum())
                fn = int((~p & g).sum())
                npos = int(g.sum())
                k = (cls, an)
                d = stats.setdefault(k, dict(tp=0, fp=0, fn=0, pos=0,
                                             n_pred=int(p.sum()), n=0))
                d['tp'] += tp; d['fp'] += fp; d['fn'] += fn
                d['pos'] += npos; d['n_pred'] += int(p.sum()); d['n'] += 1
                if npos > 0:
                    ious.append(tp / max(tp + fp + fn, 1))
            else:
                k = (cls, an)
                invalid[k] = invalid.get(k, 0) + int(p.sum())
                pass
            # 新类：GT 动作 a 的点被预测成动作 b（b != a）
        if cls in novel:
            cm = (gtm.astype(np.float32) @ pred.T.astype(np.float32))
            for ga in range(len(affs)):
                for pa in range(len(affs)):
                    if ga != pa and cm[ga, pa] > 0:
                        k = (affs[ga], affs[pa])
                        conf[k] = conf.get(k, 0) + int(cm[ga, pa])
        if ious and cls in novel:
            per_sample.append((float(np.mean(ious)), i, cls, shape_id,
                               sorted(valid_affs)))
        if (i + 1) % 200 == 0:
            print('  %d / %d' % (i + 1, len(ds)), flush=True)

    # ---- 汇总 ----
    rows = []
    for (cls, an), d in sorted(stats.items()):
        rec = d['tp'] / max(d['tp'] + d['fn'], 1)
        prec = d['tp'] / max(d['tp'] + d['fp'], 1)
        rows.append(dict(cls=cls, aff=an, split=('novel' if cls in novel else 'base'),
                         tp=d['tp'], fp=d['fp'], fn=d['fn'], pos=d['pos'],
                         n_pred=d['n_pred'], recall=rec, precision=prec,
                         overflow=d['fp'] / max(d['pos'], 1)))
    json.dump(rows, open(os.path.join(args.out, 'per_pair_errors.json'), 'w'),
              indent=1)

    inv_rows = []
    for (cls, an), v in sorted(invalid.items(), key=lambda t: -t[1]):
        inv_rows.append(dict(cls=cls, aff=an,
                             split=('novel' if cls in novel else 'base'),
                             pred_pos_points=v,
                             valid=(an in validity.get(cls, set()))))
    json.dump(inv_rows, open(os.path.join(args.out, 'invalid_pair_errors.json'), 'w'),
              indent=1)

    conf_rows = [dict(gt=ga, pred=pa, points=v)
                 for (ga, pa), v in sorted(conf.items(), key=lambda t: -t[1])]
    json.dump(conf_rows, open(os.path.join(args.out, 'action_confusion.json'), 'w'),
              indent=1)

    # 最差样本
    per_sample.sort()
    worst = per_sample[:args.topk_worst]
    payload = {}
    worst_meta = []
    for s in worst:
        coord, _, target, shape_id, cls = ds[s[1]]
        x = torch.from_numpy(coord).float().cuda().unsqueeze(0).transpose(1, 2)
        with torch.inference_mode():
            prob = torch.sigmoid(pair(x))[0].cpu().numpy()
        gt = np.asarray(target, dtype=np.float32).T
        payload['coord_%d' % s[1]] = coord
        payload['gt_%d' % s[1]] = gt
        payload['pred_%d' % s[1]] = prob
        worst_meta.append(dict(iou=s[0], idx=s[1], cls=s[2],
                               shape_id=s[3], valid_affs=s[4]))
    payload['meta'] = np.frombuffer(
        json.dumps(worst_meta).encode('utf-8'), dtype=np.uint8)
    np.savez(os.path.join(args.out, 'worst_samples.npz'), **payload)
    print('\n=== 有效配对：召回 vs 溢出（按类分组）===')
    for split in ('novel', 'base'):
        sub = [r for r in rows if r['split'] == split]
        tp = sum(r['tp'] for r in sub); fp = sum(r['fp'] for r in sub)
        fn = sum(r['fn'] for r in sub); pos = sum(r['pos'] for r in sub)
        print('%-6s pairs=%3d  TP=%7d FP=%7d FN=%7d  召回=%.3f  溢出(FP/pos)=%.3f'
              % (split, len(sub), tp, fp, fn, tp / max(tp + fn, 1), fp / max(pos, 1)))

    print('\n=== 无效配对误报 Top10 ===')
    for r in inv_rows[:10]:
        print('%-18s %-12s %-6s pred_pos=%7d' % (r['cls'], r['aff'], r['split'],
                                                 r['pred_pos_points']))
    tot_inv = sum(r['pred_pos_points'] for r in inv_rows)
    tot_fp = sum(r['fp'] for r in rows)
    print('无效配对总误报点 %d，有效配对总FP %d，占比 %.1f%%'
          % (tot_inv, tot_fp, 100 * tot_inv / max(tot_inv + tot_fp, 1)))

    print('\n=== 新类动作混淆 Top10 ===')
    for r in conf_rows[:10]:
        print('GT %-12s -> PRED %-12s %7d' % (r['gt'], r['pred'], r['points']))

    print('\n=== 最差样本 ===')
    for s in worst:
        print('IoU=%.4f idx=%d cls=%s shape=%s' % (s[0], s[1], s[2], s[3]))
    print('\n->', args.out)


if __name__ == '__main__':
    main()
