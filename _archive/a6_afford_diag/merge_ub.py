"""合并 4 个 shard 的 upper_bound 结果。"""
import glob
import json
import os
import sys

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
CASES = ['Baseline', 'Baseline-2s', 'Naive-B', 'Oracle-B', 'Oracle-B+C']


def main():
    files = sorted(glob.glob(os.path.join(CACHE, 'ub_s*.json')))
    files = [f for f in files if '.partial' not in f]
    if not files:
        raise SystemExit('no shard files')
    ious = {k: [] for k in CASES}
    cm = {k: {c: [0, 0, 0] for c in NEW_IDS} for k in CASES}
    n_bsp = {k: [] for k in CASES}
    n_ep = {k: 0 for k in CASES}

    for f in files:
        d = json.loads(open(f).read())
        for k in CASES:
            if k not in d:
                continue
            ious[k].extend(d[k]['arith'] if isinstance(d[k]['arith'], list) else [])
            n_ep[k] += d[k].get('n_ep', 0)
            for c, v in d[k]['per_class'].items():
                pass
        # shard 文件只有汇总，没有原始 ious / cm；用 arith/cumul 加权合并不可行
        print('shard', os.path.basename(f), 'ep', d['Baseline'].get('n_ep'))

    print('\n注意：shard 文件只保存了汇总指标，不能精确合并 Arith。')
    print('请改用 partial 文件（每 20 个 episode 写一次）。')
    for f in sorted(glob.glob(os.path.join(CACHE, 'ub_s*.json.partial'))):
        d = json.loads(open(f).read())
        print(os.path.basename(f), 'done', d.get('done'))


if __name__ == '__main__':
    main()
