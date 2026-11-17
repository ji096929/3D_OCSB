"""构造 23 类联合（全监督）训练的 manifest。

复刻 dataset.object_incremental.make_manifest 在 base_cap=None、index=0 时的 session 构造，
但允许只有一个 session（原实现强制 >=2 个 session）。

只读 /nas_data/ljy，输出写到 /data01/jh/afford_diag/joint23/。
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/nas_data/ljy/AffordanceNet-main')
sys.path.insert(0, str(ROOT))

from dataset.object_incremental import scan_source, file_sha256  # noqa: E402

DATA = Path('/nas_data/ljy/all_data/3D_AffordanceNet/full_shape')
OUT = Path('/data01/jh/afford_diag/joint23/manifest_all23_seed1.json')


def references(rows):
    return [{k: row[k] for k in ('index', 'shape_id', 'object_class', 'view_id') if k in row}
            for row in rows]


def main():
    train, affs = scan_source(DATA / 'full_shape_train_data.pkl')
    val, val_affs = scan_source(DATA / 'full_shape_val_data.pkl', with_validity=False)
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

    selected = list(train)
    query = list(val)
    overlap = {r['shape_id'] for r in train} & {r['shape_id'] for r in val}
    if overlap:
        raise ValueError('train/val shape_id overlap: %d' % len(overlap))

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
        'name': 'object_incremental_all23_joint',
        'seed': 1,
        'shots': 5,
        'affordances': affs,
        'query_split': 'val',
        'memory_budget': 0,
        'validity_source': 'union of positive (>0) annotations in the original training split; protocol metadata only',
        'validity': {c: [a for a in affs if a in validity[c]] for c in classes},
        'shared_affordances': list(affs),
        'sessions': [session],
        'audit': {
            'train_records': len(train),
            'val_records': len(val),
            'unique_train_instances': len({r['shape_id'] for r in train}),
            'unique_val_instances': len({r['shape_id'] for r in val}),
            'train_val_overlap': len(overlap),
            'base_affordance_coverage': len(covered),
        },
        'base_sampling': {
            'max_instances_per_class': None,
            'selection': 'all training instances; joint supervised ceiling',
            'available_instances_per_class': dict(Counter(r['object_class'] for r in train)),
            'selected_instances_per_class': dict(Counter(r['object_class'] for r in selected)),
            'selected_affordance_coverage': len(covered),
            'discarded_instances': 0,
            'validation': 'complete original val subset; no subsampling',
        },
        'sources': {split: {'filename': path.name, 'bytes': path.stat().st_size,
                            'sha256': file_sha256(path)}
                    for split, path in (('train', DATA / 'full_shape_train_data.pkl'),
                                        ('val', DATA / 'full_shape_val_data.pkl'))},
        'support_sampling': {
            'action_coverage_required': False,
            'positive_threshold': '>0',
            'source': 'not used for joint training',
            'validation_labels_used': False,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1) + '\n')
    print('classes', len(classes), 'train', len(selected), 'val', len(query),
          'valid_pairs', session['valid_pair_count'], '->', OUT)
    print('train_counts', session['train_counts'])


if __name__ == '__main__':
    main()
