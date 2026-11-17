"""Knife-stab 饱和归因：是 3D 特征本身不可分，还是属性映射溢出。

对指定 query：
  1. 取 PointNet++ 的逐点特征 h
  2. 取 GT 点 vs 高分区（pred>=0.99）的 h 余弦相似度
  3. 取解码中间量 z（grounding_head 之前）的分布
如果 h 的相似度接近 1，说明冻结骨干在刀身/刀尖上本来就分不开；
如果 h 有区分度但 z 仍然饱和，说明是属性/门控映射把差异抹平或放大。
"""
import json
import os
import sys
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

TARGET_PIDS = ['7654eded', '68458be0', 'ce23d029']  # Knife-stab


def main():
    import torch
    import torch.nn.functional as F
    from infer_graph import load_model
    from fbox.data import read, resolve, target_tensor, clear_assets
    from continual.io import image336, DATA_ROOT

    torch.set_num_threads(4)
    rows = {r['pair_id']: r for r in read('query_all.jsonl')}
    model = load_model(BASE, EXP, 'cuda')

    for pid_prefix in TARGET_PIDS:
        pid = next(k for k in rows if k.startswith(pid_prefix))
        r = rows[pid]
        xyz, tgt = target_tensor(str(resolve(r['pointcloud_path'], DATA_ROOT)),
                                 r['target_column'])         # [N,3], [N,?]
        g = (tgt.reshape(-1) > 0.5).numpy()
        img, box = image336(r)
        e = model.event(img.unsqueeze(0).cuda(), box.unsqueeze(0).cuda())   # [1,192]
        h = model.point(xyz.unsqueeze(0).cuda())                            # [1,N,192]
        logits, aux = model.decode(e, h)
        prob = logits.sigmoid()[0, :, 0].detach().cpu().numpy()

        # decode 只有一个输出通道：z 是 [B,N,192]，logits 是 [B,N,1]
        z = logits[0, :, 0].detach().cpu().numpy()
        alpha = aux['alpha'][0].detach().cpu().numpy()   # [m_total]

        hi = prob >= 0.99
        print('\n=== %s (%s-%s) ===' % (pid[:8], r['object_category'], r['action']))
        print('  N=%d  GT+=%d  high(pred>=0.99)=%d  prob p50=%.3f p90=%.3f p99=%.3f'
              % (len(prob), int(g.sum()), int(hi.sum()), np.median(prob),
                 np.percentile(prob, 90), np.percentile(prob, 99)))
        print('  logit: p10 %.3f p50 %.3f p90 %.3f p99 %.3f  max %.3f'
              % tuple([float(np.percentile(z, q)) for q in (10, 50, 90, 99)] + [float(z.max())]))
        print('  alpha(最后 stage4 的 3 个节点): %s' % np.round(alpha[-3:], 4).tolist())

        # 高分区 vs 全体的 h 相似度
        hh = F.normalize(h[0], dim=-1)
        sim = hh @ hh.T                                                     # [N,N]
        if g.sum() > 1 and (~g).sum() > 1:
            gt2hi = sim[g][:, hi].mean() if hi.sum() else float('nan')
            gt2rest = sim[g][:, ~g].mean()
            print('  h 余弦：GT+ ↔ 高分区 %.4f   GT+ ↔ 其余 %.4f   （接近 1 表示骨干特征本身分不开）'
                  % (float(gt2hi), float(gt2rest)))
        if hi.sum() > 1 and (~hi).sum() > 1:
            inner = sim[hi][:, hi][~np.eye(int(hi.sum()), dtype=bool)].mean()
            cross = sim[hi][:, ~hi].mean()
            print('  h 余弦：高分区内部 %.4f  高分区↔其余 %.4f' % (float(inner), float(cross)))
        if g.sum() > 1:
            print('  logit 在 GT+：mean %.3f  std %.3f  min %.3f  max %.3f'
                  % (float(z[g].mean()), float(z[g].std()), float(z[g].min()), float(z[g].max())))
        # z 的饱和程度：高分区内部的标准差
        if hi.sum() > 1:
            print('  logit 在高分区：mean %.3f  std %.3f  min %.3f  max %.3f'
                  % (float(z[hi].mean()), float(z[hi].std()), float(z[hi].min()), float(z[hi].max())))
            rest = ~hi
            if rest.sum():
                print('  logit 在其余：mean %.3f  std %.3f' % (float(z[rest].mean()), float(z[rest].std())))
        clear_assets()


if __name__ == '__main__':
    main()
