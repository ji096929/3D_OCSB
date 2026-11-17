"""评估全监督（23 类联合训练）DGCNN base：与 APGraph 的 38.60 / 33.76 同口径。

口径：官方 58 个有效 (object, affordance) 对，每对先对实例平均，再对配对做无权重宏平均。
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = '/nas_data/ljy/AffordanceNet-main'
sys.path.insert(0, ROOT)

DATA_ROOT = '/nas_data/ljy/all_data/3D_AffordanceNet/full_shape'
MANIFEST = '/data01/jh/afford_diag/joint23/manifest_all23_seed1.json'
# 11 base + 12 novel 的原始划分（all23 manifest 只有一个 session，不能用来分组）
SPLIT = '/nas_data/ljy/AffordanceNet-main/archive/coverage_protocol_20261004/manifests/k5_seed1.json'
OUT = '/data01/jh/afford_diag/joint23/joint_metrics.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='/data01/jh/afford_diag/joint23/base/base.pt')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--manifest', default=MANIFEST)
    ap.add_argument('--gpu', default='2')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--split', default='all23')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from attribute_graph.base import load_base
    from dataset.object_incremental import ObjectIncrementalDataset

    torch.set_num_threads(4)
    man = json.loads(open(args.manifest).read())
    affs = man['affordances']
    validity = {k: set(v) for k, v in man['validity'].items()}
    split_man = json.loads(open(SPLIT).read())
    base_cls = set(split_man['sessions'][0]['new_classes'])
    novel_cls = set(validity) - base_cls

    model, payload = load_base(args.checkpoint, 'cuda')
    model.eval()

    ds = ObjectIncrementalDataset(args.data_root, man, 0, 'val', all_objects=True)
    print('samples', len(ds), 'affs', len(affs), 'attributes', payload.get('attribute_count'), flush=True)

    buckets = defaultdict(lambda: [[], []])
    inv_pred = inv_neg = val_pred = val_neg = 0

    for i in range(len(ds)):
        coord, _, target, shape_id, cls = ds[i]
        x = torch.from_numpy(coord).float().cuda().unsqueeze(0).transpose(1, 2)  # [1,3,N]
        with torch.inference_mode():
            logits = model(x)[0]              # [18,N]（GraphModel.forward 已 transpose）
            prob = torch.sigmoid(logits).cpu().numpy()  # [18,N]
        gt = np.asarray(target, dtype=np.float32).T  # [18,N]
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
        if (i + 1) % 500 == 0:
            print('  %d / %d' % (i + 1, len(ds)), flush=True)

    eff = {k: v for k, v in buckets.items() if k[1] in validity.get(k[0], set())}

    def pair_metrics(prob_list, gt_list):
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
            p = prob > 0.5
            ious.append(float((p & g).sum() / max((p | g).sum(), 1)))
        if not aps:
            return None
        return dict(ap=float(np.mean(aps)),
                    auc=float(np.mean(aucs)) if aucs else float('nan'),
                    aiou=float(np.mean(ious)), n_inst=len(aps))

    def summarize(scope_set=None):
        items = []
        for (c, an), (pl, gl) in eff.items():
            if scope_set is not None and c not in scope_set:
                continue
            m = pair_metrics(pl, gl)
            if m is not None:
                items.append((c, an, m))
        if not items:
            return None
        return dict(pairs=len(items),
                    mAP=float(np.mean([m['ap'] for _, _, m in items])),
                    mAUC=float(np.nanmean([m['auc'] for _, _, m in items])),
                    maIOU=float(np.mean([m['aiou'] for _, _, m in items])))

    res = dict(checkpoint=args.checkpoint, samples=len(ds), split=args.split,
               invalid_pair_FPR=inv_pred / max(inv_neg, 1),
               valid_pair_FPR=val_pred / max(val_neg, 1),
               all=summarize(), base=summarize(base_cls), novel=summarize(novel_cls))
    json.dump(res, open(args.out, 'w'), indent=1)

    print('\n=== 全监督 23 类 base：官方 58 对口径 ===')
    for k in ('all', 'base', 'novel'):
        r = res[k]
        if r:
            print('%-6s pairs=%2d  mAP %6.2f  maIOU %6.2f  mAUC %6.2f'
                  % (k, r['pairs'], 100 * r['mAP'], 100 * r['maIOU'], 100 * r['mAUC']))
    print('invalid_pair_FPR %.4f  valid_pair_FPR %.4f'
          % (res['invalid_pair_FPR'], res['valid_pair_FPR']))
    print('->', args.out)


if __name__ == '__main__':
    main()
