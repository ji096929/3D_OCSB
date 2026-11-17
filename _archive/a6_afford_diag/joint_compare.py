"""汇总：11 类 base / APGraph 增量后 / 23 类全监督 在三组口径上的对照。

所有数字都来自 58 个有效对的 per-pair macro（与 AFFORD_BASELINE_DIAG 的 38.60 同口径）。
"""
import json
import os
from pathlib import Path

D = Path('/data01/jh/afford_diag')
FILES = {
    'base11_full': (D / 'base11_on23.json', None),
    'apgraph_full': (D / 'oracle_full_k5.json', 'raw'),
    'joint_full': (D / 'joint23/joint_metrics.json', None),
    'apgraph_partial': (D / 'oracle_partial_k5.json', 'raw'),
    'joint_partial': (D / 'joint23_partial/joint_metrics.json', None),
}


def load(path, key):
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    return d[key] if key else d


def row(name, d, scope):
    if d is None:
        return None
    if scope == 'base':
        # APGraph 的 base 部分被冻结，与 11 类 base 相同
        src = d if 'base' in d else None
        if src is None:
            return None
    return d.get(scope)


def main():
    data = {k: load(p, s) for k, (p, s) in FILES.items()}

    print('%-22s %9s %9s %9s %9s %9s' %
          ('model', 'All mAP', 'All maIOU', 'All mAUC', 'Base mAP', 'Novel mAP'))
    for name in ('base11_full', 'apgraph_full', 'joint_full',
                 'apgraph_partial', 'joint_partial'):
        d = data[name]
        if d is None:
            print('%-22s  (missing)' % name)
            continue
        a = d.get('all') or {}
        b = d.get('base') or {}
        n = d.get('novel') or {}
        print('%-22s %9.2f %9.2f %9.2f %9s %9s' % (
            name,
            100 * a.get('mAP', float('nan')), 100 * a.get('maIOU', float('nan')),
            100 * a.get('mAUC', float('nan')),
            '%.2f' % (100 * b['mAP']) if b.get('mAP') else '--',
            '%.2f' % (100 * n['mAP']) if n.get('mAP') else '--'))

    # 分解：全监督 gap
    print('\n=== 分解（full 协议，58 对口径）===')
    base11 = (data.get('base11_full') or {}).get('all', {})
    apg = (data.get('apgraph_full') or {}).get('all', {})
    jt = (data.get('joint_full') or {}).get('all', {})
    if base11 and apg:
        print('增量增益（APGraph - 11类base）: mAP %+.2f  maIOU %+.2f  mAUC %+.2f'
              % (100 * (apg['mAP'] - base11['mAP']), 100 * (apg['maIOU'] - base11['maIOU']),
                 100 * (apg['mAUC'] - base11['mAUC'])))
    if base11 and jt:
        print('全监督增益（joint - 11类base）: mAP %+.2f  maIOU %+.2f  mAUC %+.2f'
              % (100 * (jt['mAP'] - base11['mAP']), 100 * (jt['maIOU'] - base11['maIOU']),
                 100 * (jt['mAUC'] - base11['mAUC'])))
    if jt and apg:
        print('APGraph 仍差（joint - APGraph）  : mAP %+.2f  maIOU %+.2f  mAUC %+.2f'
              % (100 * (jt['mAP'] - apg['mAP']), 100 * (jt['maIOU'] - apg['maIOU']),
                 100 * (jt['mAUC'] - apg['mAUC'])))
        print('  → 这部分是「少样本 + 增量顺序」造成的损失')


if __name__ == '__main__':
    main()
