# 零泄漏训练集重建与 Step 5 复检

日期：2026-11-13
目的：解决"训练任务与验证任务场景重叠 82%"的泄漏问题，并把 Step 5 的训练规模从
200 任务 × 8 epoch（1600 步）扩到 2000 任务 × 2 epoch（4000 步）。

---

## 1. 泄漏发现

| 检查项 | 结果 |
|---|---|
| `blocks_bs1_s1/data` 场景数 | 312 |
| 其中 train split（1201 场景） | **0** |
| 其中 val split | **312** |
| `train_200` 的 support 场景 | 135，100% 在 val split |
| `train_200` 与 `val_560` 场景重叠 | **154 / 187（82%）** |

COSeg 的预处理只发布了 val 的 blocks。旧的 Step 5 训练任务全部来自 val 场景，
与验证集有 82% 的场景重叠。

---

## 2. 修复流程

1. `scannet_raw/scans` 包含全部 1201 个 train 场景（ply + segs.json + aggregation.json）
2. `collect_train_blocks.py` 取前 400 个 train 场景，生成 8 列 npy（xyz+rgb+label+seg）
3. `room2blocks.py` 切成 **9425 个 1m block**
4. `build_train2k.py` 采 **2000 个零泄漏训练任务**（10 个 base 类 × 200，跨场景）

**踩坑**：`get_raw2scannet_label_map` 依赖模块全局 `CLASS_NAMES`。如果先调用它、
再设置 `CLASS_NAMES`，所有 `nyu40_name` 都会映射成 "unannotated"（label 0）。
第一轮 400 个场景因此全是 label 0，修复顺序后重跑。

**隔离性**：train split 的 400 个场景与 val_560 的 260 个 val 场景完全不相交，
所以 `val_560` 缓存直接复用。

---

## 3. Step 1–4（val_560，零泄漏）

| 版本 | γ=1.5 | γ=1.75 | γ=2.0 |
|---|---:|---:|---:|
| Step 1 COSeg | 0.3744 | 0.3744 | 0.3744 |
| Step 2 直接扩展 B | 0.3717 | 0.3709 | 0.3702 |
| Step 3 OASB | 0.3772 | 0.3768 | 0.3763 |
| Step 4 OASB+CSR | **0.3773** | 0.3770 | 0.3761 |

（Cumul / MissRec / FalseRec / Cprec / AUC 见 `ablation_full_g*.json`）

---

## 4. Step 5（2000 任务 × 2 epoch = 4000 步）

待填。

---

## 5. death_test 复检

待填。

---

## 6. 结论

待填。
