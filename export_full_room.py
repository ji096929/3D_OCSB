"""导出 3 个典型任务对应的完整 ScanNet 房间（真彩 + 切块高亮）。

对每个 U block：
  1. 读完整房间 <scene>_vh_clean_2.ply（场景坐标、真彩）
  2. 读 block 的 .npy（同为场景坐标）取其 bbox
  3. 导出两套：
       <task>_<block>_full_rgb.ply   完整房间真彩
       <task>_<block>_full_box.ply   切块内点染亮黄，其余真彩压暗
"""
import json
import os

import numpy as np
from plyfile import PlyData

CACHE = '/data01/jh/data/ocsb_cache'
RAW = '/data01/jh/data/scannet_raw/scans'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
OUT = os.path.join(CACHE, 'ply_full')
TASKS = (7, 9, 11)


def read_ply_xyzrgb(path):
    ply = PlyData.read(path)
    v = ply['vertex']
    xyz = np.stack([np.asarray(v['x']), np.asarray(v['y']),
                    np.asarray(v['z'])], 1).astype(np.float64)
    rgb = np.stack([np.asarray(v['red']), np.asarray(v['green']),
                    np.asarray(v['blue'])], 1).astype(np.float64)
    return xyz, rgb


def write_ply(path, xyz, rgb):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write('element vertex %d\n' % len(xyz))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\n'
                'property uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyz)):
            f.write('%.4f %.4f %.4f %d %d %d\n'
                    % (xyz[i, 0], xyz[i, 1], xyz[i, 2],
                       int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])))


def main():
    os.makedirs(OUT, exist_ok=True)
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    cache = {}
    index = []
    for idx in TASKS:
        m = next(x for x in metas if x['idx'] == idx)
        for nm in m['u']:
            scene = nm.split('_block_')[0]
            if scene not in cache:
                p = os.path.join(RAW, scene, scene + '_vh_clean_2.ply')
                cache[scene] = read_ply_xyzrgb(p)
                print('loaded room', scene,
                      len(cache[scene][0]), flush=True)
            xyz, rgb = cache[scene]
            b = np.load(os.path.join(DATA, nm + '.npy'))
            bx = b[:, 0:3].astype(np.float64)
            mn, mx = bx.min(0), bx.max(0)
            inside = ((xyz[:, 0] >= mn[0] - 0.02) &
                      (xyz[:, 0] <= mx[0] + 0.02) &
                      (xyz[:, 1] >= mn[1] - 0.02) &
                      (xyz[:, 1] <= mx[1] + 0.02) &
                      (xyz[:, 2] >= mn[2] - 0.02) &
                      (xyz[:, 2] <= mx[2] + 0.02))
            print('  %s  block pts %d  room pts inside %d  bbox %s -> %s'
                  % (nm, len(bx), int(inside.sum()),
                     np.round(mn, 2), np.round(mx, 2)), flush=True)

            base = 'task%02d_%s' % (idx, nm)
            write_ply(os.path.join(OUT, base + '_full_rgb.ply'), xyz, rgb)
            box_rgb = rgb * 0.45
            box_rgb[inside] = [255, 230, 0]
            write_ply(os.path.join(OUT, base + '_full_box.ply'), xyz, box_rgb)
            index.append({'task': idx, 'class': m['class'], 'block': nm,
                          'scene': scene, 'n_room': int(len(xyz)),
                          'n_block': int(inside.sum()),
                          'bbox_min': mn.tolist(), 'bbox_max': mx.tolist()})
    json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
    print('exported', len(index), '->', OUT)


if __name__ == '__main__':
    main()
