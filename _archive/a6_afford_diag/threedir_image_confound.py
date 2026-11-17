"""3DIR：功能框大小 / 图像可用性 与逐对指标的关联。

如果"人手遮挡或图文姿态不一致"是主因，功能框（人手所在区域）越小，指标应该越差。
对 APGraph（不用图像）和 IAGNet/GREAT（用图像）分别算 Spearman 相关做对照。
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

PROTO = Path('/nas_data/ljy/baseline/3DIR_Order2143_20261109/protocol/query_all.jsonl')
RUNS = {
    'IAGNet': ('/nas_data/ljy/baseline/IAGNet_DINO_HardLabel_ES20/runs/3dir/'
               '3dir_swap_grasp_wrapgrasp_with_lift_pour_v1', 'evaluations/stage4_query_pairs.jsonl'),
    'GREAT': ('/nas_data/ljy/baseline/GREAT_HardLabel_ES20/runs/3dir/'
              '3dir_swap_grasp_wrapgrasp_with_lift_pour_v1', 'evaluations/stage4_query_pairs.jsonl'),
}
SEEDS = range(5)


def main():
    rows = [json.loads(l) for l in PROTO.read_text().splitlines() if l.strip()]
    meta = {r['pair_id']: r for r in rows}

    sizes = {}
    for r in rows:
        p = r['image_path']
        if p not in sizes:
            with Image.open(p) as im:
                sizes[p] = im.size
    for r in rows:
        w, h = sizes[r['image_path']]
        x1, y1, x2, y2 = r['functional_bbox_xyxy']
        r['img_w'], r['img_h'] = w, h
        r['bbox_ratio'] = max((x2 - x1) * (y2 - y1), 1) / float(w * h)
        r['bbox_side'] = float(np.sqrt(max((x2 - x1) * (y2 - y1), 1)))

    ratios = np.array([r['bbox_ratio'] for r in rows])
    print('功能框面积占比: min %.4f  p25 %.4f  median %.4f  p75 %.4f  max %.4f'
          % (ratios.min(), np.percentile(ratios, 25), np.median(ratios),
             np.percentile(ratios, 75), ratios.max()))
    print('图像尺寸分布:', {s: sum(1 for v in sizes.values() if v == s) for s in set(sizes.values())})

    for name, (root, rel) in RUNS.items():
        per_pid = {}
        for seed in SEEDS:
            f = Path(root) / ('K5_seed%d' % seed) / rel
            if not f.is_file():
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                pid = rec.get('pair_id')
                if pid in meta:
                    per_pid.setdefault(pid, []).append(rec)
        if not per_pid:
            print(name, 'no pairs'); continue
        aucs, aious, br, bs = [], [], [], []
        for pid, lst in per_pid.items():
            aucs.append(np.mean([x['AUC'] for x in lst]))
            aious.append(np.mean([x['aIOU'] for x in lst]))
            br.append(meta[pid]['bbox_ratio'])
            bs.append(meta[pid]['bbox_side'])
        r_auc, p_auc = spearmanr(br, aucs)
        r_iou, p_iou = spearmanr(br, aious)
        r2_auc, p2_auc = spearmanr(bs, aucs)
        print('%s  n=%d  AUC vs bbox面积占比 rho=%.3f (p=%.3g)   aIOU vs bbox面积占比 rho=%.3f (p=%.3g)'
              % (name, len(aucs), r_auc, p_auc, r_iou, p_iou))
        # 按 bbox 大小四分位分组
        q = np.percentile(br, [25, 50, 75])
        groups = {0: [], 1: [], 2: [], 3: []}
        for x, b in zip(aucs, br):
            groups[0 if b <= q[0] else 1 if b <= q[1] else 2 if b <= q[2] else 3].append(x)
        print('   AUC by bbox 面积四分位（小→大）:',
              ['%.2f (n=%d)' % (np.mean(g), len(g)) for k, g in groups.items()])

    # 动作层面的对照：每个动作的 bbox 平均大小 vs 方法在该动作的 AUC
    print('\n逐动作：bbox 面积占比中位数 vs APGraph/IAGNet 的 AUC')
    agg = {}
    for r in rows:
        agg.setdefault(r['action'], []).append(r['bbox_ratio'])
    for a in sorted(agg):
        print('  %-10s bbox_med %.4f' % (a, float(np.median(agg[a]))))


if __name__ == '__main__':
    main()
