"""实验 2：或门直接恢复（OR-Gate Recovery）。

保留 q_margin > 0 的全部预测，在 Q_bg_pred 的头部超点里直接翻转。
扫 (score, Top-K%, tau_deep)。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_bc as D  # noqa: E402
from ocsb_cmc import OCSBCMC  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'
SCORES = ('margin', 'B_max', 'mean')
KS = (0.01, 0.02, 0.05)


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--out', default='/data01/jh/data/ocsb_cache/or_gate.json')
    args = ap.parse_args()

    torch.manual_seed(0)
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()
    cfg = D.load_cfg_from_cfg_file(D.CFG_PATH)
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:args.n]
    print('episodes:', len(metas), flush=True)

    qprep, qtree = {}, {}
    blocks = {}
    for m in metas:
        name = m['query']
        if name not in blocks:
            blocks[name] = np.load(os.path.join(DATA, name + '.npy'))
    print('blocks:', len(blocks), flush=True)

    cmc = None
    # 统计每个配置的 per-episode IoU
    acc = {}
    base_ious = []
    for ei, m in enumerate(metas):
        d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        ep = {k: to_t(d[k]) for k in d.files}
        ep['base_protos'] = table
        m['y_full'] = d['q_y_full'].astype(np.int64)
        if cmc is None:
            cmc = OCSBCMC(int(ep['s_h'].shape[1]), tau=0.1, tau_b=0.6,
                          m_anchor=3).cuda()
        with torch.no_grad():
            out = cmc(ep, use_csr=True)
        z_q, M, w = out['z_q'], out['M_feat'], out['M_w']
        margin = ep['q_margin']
        scores = {'margin': margin}
        if M.shape[0] > 0:
            sim = z_q @ M.t()
            wh = w / (w.sum() + 1e-6)
            scores['mean'] = (wh.unsqueeze(0) * sim).sum(1)
        else:
            scores['mean'] = torch.zeros_like(margin)
        if out['z_B'].shape[0] > 0:
            scores['B_max'] = (out['q_B'].unsqueeze(0)
                               * (z_q @ out['z_B'].t())).max(1).values
        else:
            scores['B_max'] = torch.zeros_like(margin)

        # query 超点
        name = m['query']
        if name not in qprep:
            b = blocks[name]
            qprep[name] = D.data_prepare_seg(
                b[:, 0:3], b[:, 3:6], b[:, 6], b[:, 7],
                cfg.voxel_size, cfg.voxel_max)
        q_coord_t, _, _, q_seg_t = qprep[name]
        if name not in qtree:
            qtree[name] = cKDTree(q_coord_t.numpy().astype(np.float64))
        idx = qtree[name].query(
            ep['q_coord'].cpu().numpy().astype(np.float64))[1]
        q_seg = q_seg_t.numpy()[idx]
        uniq, inv = np.unique(q_seg, return_inverse=True)
        inv_t = torch.from_numpy(inv).cuda()
        n_sp = len(uniq)
        cnt = torch.bincount(inv_t, minlength=n_sp).clamp_min(1).float()
        m_sp = torch.zeros(n_sp, device=margin.device)
        m_sp.index_add_(0, inv_t, margin)
        m_sp = m_sp / cnt
        bg_sp = m_sp <= 0
        n_bg = int(bg_sp.sum())
        if n_bg == 0:
            continue

        tree_q = cKDTree(ep['q_coord'].cpu().numpy().astype(np.float64))
        dd, ii = tree_q.query(
            ep['q_coord_full'].cpu().numpy().astype(np.float64), k=3)
        ww = 1.0 / (dd + 1e-8)
        ww /= ww.sum(1, keepdims=True)

        def iou_of(ell):
            v = (ell.detach().cpu().numpy().astype(np.float64)[ii] * ww).sum(1)
            y = m['y_full']
            p = v > 0
            tp = int((p & (y > 0)).sum())
            un = int((p | (y > 0)).sum())
            return tp / max(un, 1)

        base_ious.append(iou_of(margin))
        for sn in SCORES:
            s_sp = torch.zeros(n_sp, device=margin.device)
            s_sp.index_add_(0, inv_t, scores[sn])
            s_sp = s_sp / cnt
            ids = bg_sp.nonzero().flatten()
            order = torch.argsort(-s_sp[ids])
            for k in KS:
                kk = max(int(n_bg * k), 1)
                keep = ids[order[:kk]]
                mask = torch.zeros(n_sp, dtype=torch.bool, device=margin.device)
                mask[keep] = True
                sel = mask[inv_t]
                ell = margin.clone()
                ell[sel] = 1.0
                key = '%s_k%.2f' % (sn, k)
                acc.setdefault(key, []).append(iou_of(ell))
        if (ei + 1) % 50 == 0:
            print('  %d / %d' % (ei + 1, len(metas)), flush=True)

    base = np.array(base_ious)
    print('\nbaseline arith %.4f' % base.mean())
    res = {'baseline': {'arith': float(base.mean())}}
    for key in sorted(acc):
        v = np.array(acc[key])
        d = v - base
        pse = d.std(ddof=1) / np.sqrt(len(d))
        res[key] = {'arith': float(v.mean()), 'delta': float(d.mean()),
                    'pse': float(pse),
                    't': float(d.mean() / pse) if pse else 0.0}
        print('%-14s arith %.4f  d %+.4f  pSE %.4f  t %+.2f'
              % (key, v.mean(), d.mean(), pse,
                 d.mean() / pse if pse else 0))
    json.dump(res, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
