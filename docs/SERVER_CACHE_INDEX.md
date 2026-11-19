# 服务器缓存索引（/data01/jh）

> 生成：2026-11-19 19:20
> 用途：**上下文压缩后据此恢复"哪些缓存还在、能干什么、重建要多久"**，避免重复计算。
> 总量：`/data01/jh/data` 152 GB、`exp` 35 GB、日志 1.8 GB、网盘 399/1107 GB。
> **当前 GFS-VL 研究真正依赖的只有约 11 GB（标 ★），其余是 OCSB 阶段遗留或可重建产物。**

---

## 1. 输入数据

| 路径 | 大小 | 内容 | 重建 |
|---|---|---|---|
| `data/scannet200_gfs` ★★ | 12 G | GFS 数据集：train 1201 / val 312 场景，每场景 `coord/color/normal/segment200/segment20/instance.npy`；57 类映射用 `vlm_common.convert_to_gfs` | 从 ScanNet v2 重切，约 1 h |
| `data/scannet_raw` | 9.9 G | ScanNet v2 原始 `scans/`（ply + aggregation + segs.json）。**没有图像、没有位姿、没有内参** | gated 下载 |
| `data/ScanNet` + `data/scannet_train_v2` | 5.5 G + 19 G | OCSB 阶段的 block 缓存（`blocks_bs1*`） | 可重建，但已无用 |
| `data/support_protos` ★★ | 2 M | 5 个 split ×（VLM 512d、Sonata 1232d）的 1-shot 支持原型，文件名 `sc200_regis_1_10{,_sonata}.npy` 等 | `build_support_protos*.py` |
| `data/gc_full.npy` | 488 B | 1201 场景 × 45 新类的 GT 点数（mIoU-N 的分母） | 扫描 GT，几分钟 |

## 2. 区域级缓存（GFS-VL 阶段）

| 路径 | 大小 | 内容 | 重建 |
|---|---|---|---|
| `data/region_cache` ★ | 97 M | 30478 个「novel 预测类」连通分量：均值特征 512d、sizes、hist_n、hist_b、n_ign | `build_region_cache.py --shard i --shards 8`，8 卡约 20 min |
| `data/region_index` ★ | 547 M | 区域 → 点索引（`idx/off/pred`） | `dump_region_index.py` |
| `data/base_cache` ★ | 124 M | 39139 个「base 预测类」区域，同一套字段 | `build_base_cache.py` |
| `data/base_index` ★ | 940 M | base 区域 → 点索引 | `dump_base_index.py` |
| `data/sonata_region` ★ | 106 M | 与 `region_cache` 同一批区域的 Sonata 1232d 特征 | `build_sonata_region.py`，8 卡约 10 min |
| `data/sonata_base` ★ | 135 M | base 区域的 Sonata 特征（2026-11-19 补建，AI 点权重要用） | `build_sonata_base.py`，单卡约 40 min |
| `data/scene_feats_cache` | **87 G** | **训练集逐点 VLM 特征** `feats (N,512) fp16` + `gfs` 标签，每场景一个 npz | 无现成脚本，单卡 VLM 前向约 20 min；**无后续点级 VLM 需求可删** |

## 3. 点级伪标签 / 权重 / 分数

| 路径 | 大小 | 内容 | 重建 |
|---|---|---|---|
| `data/online_pl` ★★ | 37 M | **官方在线 PS 与 PS+AI 的点级伪标签**（1201 场景 npz：`ps` / `psai` int16，整场景口径） | `dump_online_pl.py`，单卡约 20 min |
| `data/online_pl_w` ★★ | 1003 M | **逐点软权重**（PS `0.25+0.75σ(2z)`、AI `0.05+0.55σ(2z)`），`_pl.npy` + `_w.npy` | `build_w_pp.py`（依赖 `pp_score`），约 3 min |
| `data/online_pl_w_uniform` | 336 M | 均匀权重版（A2 消融），`_pl.npy` 是软链 | `build_w_uniform.py`，1 min |
| `data/pp_score` | 336 M | 训练集逐点 Sonata 对其「被赋类」的余弦 | `build_pp_score.py`，单卡 24 min / 2 卡 12 min |
| `data/offline_pl` | 4.3 G | 旧版离线伪标签：`sc200_regis_1_10{,_A,_B,_SW}` + `sc200_regis_1_{20..50}_A` | `gen_ab.py` / `gen_offline_pl.py` / `gen_sw.py` |
| `data/val_sonata_cos` ★ | 4.2 G | **312 val 逐点 45 类 Sonata 余弦**（`float16[N,45]`，仲裁实验用） | `dump_val_sonata_cos.py`，2 卡约 6 min |
| `data/val_logits` ★ | 492 M | **312 val 逐点 top5**（`top5p/top5i/pred/gt`，ONLINE_s42） | `val_dump_logits2.py`，7 卡约 2 min（**并发启动要错开 6 s**） |
| `data/val_logits_s7` | 110 M | 同 s7，**只跑了 73/312 场景（23%）** | 同上 |
| `data/val_logits_s13` | 32 M | 同 s13，**只有 21/312 场景（7%）** | 同上 |
| `data/val_cos_stats*.npy` | <1 K | 类内均值/方差（校准实验） | `arbiter_v4/v5.py` 首次运行自动生成 |

## 4. 模型权重（`/data01/jh/code/GFS-VL/exp`，35 G）

| 目录 | 大小 | 说明 |
|---|---|---|
| `ptv3_bk` ★★ | 529 M | Stage-1 backbone（`epoch 81`，无 novel_seg_head） |
| `sc200_k1_regis1` ★★ | 530 M | **官方 Model Zoo 权重**（`epoch 20`，`optimizer.step 1999`） |
| `sc200_k1_ONLINE_s42 / s7 / s13` | 各 3.7 G | 原版在线基线 3 个 seed（Novel 0.2679 / 0.2765 / 0.2669） |
| `sc200_k1_A2_s42` | 3.7 G | 均匀权重消融（0.2685） |
| `sc200_k1_SW3_s42 / s7` | 各 3.7 G | 逐点软权重（0.2644 / 0.2832） |
| `sc200_k1_offline` | 4.2 G | 旧版 offline（**2 split / 40 epoch**，0.2842） |
| `sc200_k1_offline_A / _B` | 各 3.7 G | 旧版离线 A / B（0.2711 / 0.2520） |
| `sc200_k1_SW2_s42` | 76 K | **废弃**（启动发现 Collect 漏字段，已 kill） |
| 每目录内 `epoch_{3,6,9,12,15,18}.pth` | 每个 554 M | **历史副本，可全删**（只留 `*_last.pth` / `*_best.pth`），约省 20 G |

## 5. OCSB 阶段遗留（与当前研究无关，可删/归档，约 32 G）

- `data/scannet_train_v2` 19 G、`data/ScanNet` 5.5 G：block 级预处理缓存
- `data/ocsb_train2k` 4.2 G：episode 定义（support/query/U block）
- `data/ocsb_cache` 2.3 G：ablation 结果 json
- `data/retrieval_cache` 396 M：U 池特征
- `data/sonata_dump` 185 M、`data/orth_cache` 150 M
- `data/s2_main` / `s2_main2` / `s2_lr4_v100` 共 117 M：早期 checkpoint
- 几十个小目录（`diag_*` / `pm_*` / `clip_*` / `cs2` / `vs` / `vx` / `cx2` / `ab_*` / `base_*` / `cal_*` / `gate_*` / `p3*` / `group_exp` / `miss_cache` / `s2_cross`）共约 **547 M**

## 6. 日志（`/data01/jh/afford_diag`，1.8 G）

- `gfs_build/*.log` — **本轮全部实验日志**（`GFS_VL_CONCLUSION.md` 引用的行号都在这里）
  - 关键：`test_regis1_v3.log`（官方权重）、`train_ONLINE_s42.log`、`guard_sc200_k1_*.log`、`online_pl_pts.log`、`per_class_pts.log`、`cm2_s42.log`、`cm2_s7.log`、`oracle_s42.log`、`arbiter_*.log`、`ai_auc.log`
- `ocsb_logs/` — 早期 OCSB 阶段
- 其余是 ply / png 可视化产物

## 7. ScanNet++（在网盘，不在本机）

- `/scannetpp/<scene>/scans`：954 场景，**约 142 GB**，每场景 5 个文件（mesh 11–131 MB + semantic + segments + anno + mask）
- 分享里 52 个场景没有 `scans` 目录（MISS）
- **没有转存 `dslr/` 和 `iphone/`（图像）**
- 网盘：已用 399 GB / 总 1107 GB

## 8. 后台任务状态

| 任务 | 状态 |
|---|---|
| HF Pointcept 预处理包（325 GB） | **基本停滞**：`data/scannetpp_parts/part00` 现 4 MB，速率 ~1.6 KB/s；曾到 11 GB 后被截断（日志 `hf_download.log` 停在 11:43） |
| 训练任务 | 无 |
| GPU | smi 0–5、7 空闲；smi 6 被 xzy 占 28 GB |

## 9. 重建成本速查

```bash
cd /data01/jh/code/ps_diag
# 区域缓存（8 卡并行，每卡一个 shard）
python build_region_cache.py --shard $i --shards 8      # ~20 min
python build_sonata_region.py --shard $i               # ~10 min
python build_sonata_base.py --shard $i                 # 单卡 ~40 min
# 点级伪标签 / 分数
python dump_online_pl.py                               # ~20 min
python build_pp_score.py --shard $i --shards 2         # 2 卡 12 min
python dump_val_sonata_cos.py --shard $i --shards 2    # 2 卡 6 min
python val_dump_logits2.py <cfg> <pth> <out> $i 7      # 7 卡 2 min，**错开 6 s 启动**
```

## 10. 踩过的坑（避免重复）

1. **并发启动多个 python 进程读同一个 exp config** → `'ConfigDict' object has no attribute 'seed'`。错开 6 秒启动即可。
2. `default_setup(cfg)` 会往 `save_path` 写 config 快照，多进程共用同一 `save_path` 会互相覆盖 → 每个分片用独立目录。
3. 旧区域级 `S[k]` 打分（AUC 0.659）不能用；**逐点 Sonata 才是 0.783**。
4. 数据集是 **1201 场景**（不是 1195）；A/B 的伪标签只覆盖 1195，少了 5–6 个场景。
5. `train_guard.sh` 的 resume 逻辑在**多 split** 训练下会跳过后续 split（`start_epoch` 不重置），目前只对单 split 安全。
