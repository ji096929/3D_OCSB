"""构造 partial（单视角）协议的 23 类联合训练 manifest。

每个原始物体保留一个训练视角（与 full_shape 的 16082 个物体一一对应），
验证集保留全部 9140 个官方视角。

只读 /nas_data/ljy，输出写到 /data01/jh/afford_diag/joint23_partial/。
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/nas_data/ljy/AffordanceNet-main')
sys.path.insert(0, str(ROOT))

from dataset.object_incremental import scan_source, file_sha256  # noqa: E402

DATA = Path('/nas_data/ljy/all_data/3D_AffordanceNet/partial')
OUT = Path('/data01/jh/afford_diag/joint23_partial/manifest_all23_seed1.json')


def references(rows):
    return [{k: row[k] for k in ('index', 'shape_id', 'object_class', 'view_id') if k in row}
            for row in rows]


def main():
    train, affs = scan_source(DATA / 'partial_train_data.pkl', data_setting='partial')
    val, val_affs = scan_source(DATA / 'partial_val_data.pkl', with_validity=False,
                                data_setting='partial')
    if affs != val_affs:
        raise ValueError('train/val affordance order differs')

    classes = sorted({r['object_class'] for r in train})
    if {r['object_class'] for r in val} != set(classes):
        raise ValueError('val classes differ from train classes')

    validity = {c: set() for c in classes}
    for r in train:
        validity[r['object_class']].update(r['positive_affordances'])
    covered = set().union(*(validity[c] for c in classes))
    if covered != set(affs):
        raise ValueError('joint training objects miss affordances: %s' % (set(affs) - covered))

    # 每个原始物体保留一个训练视角：按 (index, view_id) 稳定取第一个
    ordered = sorted(train, key=lambda r: (r['index'], str(r.get('view_id'))))
    seen, selected = set(), []
    for r in ordered:
        if r['shape_id'] in seen:
            continue
        seen.add(r['shape_id'])
        selected.append(r)
    query = list(val)
    if {r['shape_id'] for r in selected} & {r['shape_id'] for r in query}:
        raise ValueError('train/val shape_id overlap')

    positive = Counter((r['object_class'], a) for r in selected
                       for a in r['positive_affordances'])
    session = {
        'session': 0,
        'new_classes': classes,
        'seen_classes': classes,
        'object_class_count': len(classes),
        'affordances': affs,
        'affordance_count': len(affs),
        'valid_pair_count': sum(len(validity[c]) for c in classes),
        'train_affordances': affs,
        'train_affordance_count': len(affs),
        'train_valid_pair_count': len(positive),
        'positive_train_instances_per_pair': {
            c: {a: positive[(c, a)] for a in affs if a in validity[c]} for c in classes},
        'train': references(selected),
        'query': references(query),
        'train_counts': dict(Counter(r['object_class'] for r in selected)),
        'query_counts': dict(Counter(r['object_class'] for r in query)),
    }

    manifest = {
        'schema_version': 1,
        'name': 'object_incremental_all23_joint_partial',
        'seed': 1,
        'shots': 5,
        'affordances': affs,
        'query_split': 'val',
        'memory_budget': 0,
        'data_setting': 'partial',
        'validity_source': 'union of positive (>0) annotations in the original training split; protocol metadata only',
        'validity': {c: [a for a in affs if a in validity[c]] for c in classes},
        'shared_affordances': list(affs),
        'sessions': [session],
        'audit': {
            'train_records': len(train),
            'val_records': len(val),
            'unique_train_instances': len({r['shape_id'] for r in train}),
            'unique_val_instances': len({r['shape_id'] for r in val}),
            'train_val_overlap': 0,
            'base_affordance_coverage': len(covered),
        },
        'base_sampling': {
            'max_instances_per_class': None,
            'selection': 'one training view per original object; joint supervised ceiling',
            'available_instances_per_class': dict(Counter(r['object_class'] for r in train)),
            'selected_instances_per_class': dict(Counter(r['object_class'] for r in selected)),
            'selected_affordance_coverage': len(covered),
            'discarded_instances': 0,
            'discarded_views': len(train) - len(selected),
            'validation': 'complete original val views; no subsampling',
        },
        'view_protocol': {
            'shots_unit': 'independent original shape_id; one annotated view per selected object',
            'base': 'one training view per original object',
            'query': 'all official validation views',
            'query_labels_used_for_selection': False,
        },
        'sources': {split: {'filename': path.name, 'bytes': path.stat().st_size,
                            'sha256': file_sha256(path)}
                    for split, path in (('train', DATA / 'partial_train_data.pkl'),
                                        ('val', DATA / 'partial_val_data.pkl'))},
        'support_sampling': {
            'action_coverage_required': False,
            'positive_threshold': '>0',
            'source': 'not used for joint training',
            'validation_labels_used': False,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1) + '\n')
    print('classes', len(classes), 'train_views', len(selected), 'of', len(train),
          'val_views', len(query), 'valid_pairs', session['valid_pair_count'], '->', OUT)


if __name__ == '__main__':
    main()
