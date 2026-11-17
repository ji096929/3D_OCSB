"""IAGNet 冻结双编码器的可塑性下限测试。

冻结 DINOv2 (img_encoder) + PointNet++ (point_encoder, sa1~sa3)，
只训 JRA + ARM + Decoder (fp1~fp3 + out_head)。

从 base/best.pt 开始，按 stage 1 → 2 → 3 顺序跑（与 IAGNet 的协议顺序一致：
s1=grasp/wrapgrasp, s2=lay/move, s3=cut/stab），每阶段结束出逐动作/逐对指标。

输出：/data01/jh/afford_diag/iagnet_frozen/seed{seed}/stage{n}/
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path('/nas_data/ljy/baseline/IAGNet_DINO_HardLabel_ES20')
sys.path.insert(0, str(ROOT))
# dataset_setup.activate() 在 fbox/data.py 顶部被调用，读 GROUNDING_DATASET
os.environ['GROUNDING_DATASET'] = '3dir'

BASE = ROOT / 'runs/3dir/3dir_swap_grasp_wrapgrasp_with_lift_pour_v1/base/best.pt'
OUT = Path('/data01/jh/afford_diag/iagnet_frozen')


def freeze_encoders(model):
    for mod in (model.img_encoder, model.point_encoder):
        for p in mod.parameters():
            p.requires_grad_(False)
        mod.eval()
    original_train = model.train

    def train(mode=True):
        original_train(mode)
        model.img_encoder.eval()
        model.point_encoder.eval()
        return model

    model.train = train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--shot', type=int, default=5)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--stages', default='1')
    ap.add_argument('--base', default=str(BASE))
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    import torch
    from fbox.engine import make_model, load_model, train_steps, evaluate, seed_all
    from fbox.data import support_pool, read, clear_assets
    from fbox.metrics import macro
    from run_fbox import configuration

    ns = argparse.Namespace(config=str(ROOT / 'configs/3dir.json'), seed=a.seed, shot=a.shot,
                            device=a.device, smoke=False, smoke_batch_size=16, base_run=None)
    cfg = configuration(ns)
    seed_all(a.seed)
    model = make_model(cfg, a.device)
    load_model(Path(a.base), model)
    freeze_encoders(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print('trainable %d / %d (%.1f%%)' % (n_train, n_total, 100 * n_train / n_total), flush=True)

    query = read('query_all.jsonl')
    out_root = OUT / (('seed%d' % a.seed) + (('_' + a.tag) if a.tag else ''))

    for session in [int(x) for x in a.stages.split(',')]:
        out = out_root / ('stage%d' % session)
        out.mkdir(parents=True, exist_ok=True)
        resume = out / ('incremental_session%d_last.pt' % session)
        if resume.exists():
            resume.unlink()
        pool = support_pool(a.shot, a.seed, session, a.seed + 10000 * session)
        print('session %d support budget %s' % (session, pool.budget()), flush=True)
        train_steps(model, pool, cfg, a.device, cfg['incremental_policy']['epochs'],
                    cfg['incremental_policy']['lr'], out, session)
        values = evaluate(model, query, cfg, a.device, out, 'stage%d_frozen' % session)
        if not isinstance(values, dict):
            raise TypeError('evaluate returned %r' % type(values))
        vals = {k: round(float(v['aIOU']), 2) for k, v in sorted(values.items())}
        print('stage%d frozen actions: %s' % (session, vals), flush=True)
        json.dump(values, open(out / 'stage%d_frozen_actions.json' % session, 'w'), indent=1)
        clear_assets()


if __name__ == '__main__':
    main()
