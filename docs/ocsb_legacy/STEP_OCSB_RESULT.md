# 3D-OCSB 第一步验证实验：B/C 质量统计

日期：2026-11-05
目标：冻结 COSeg（1-way 1-shot，ScanNet val），统计候选桥梁 B 的质量、
遗漏集合 C 的规模、桥接优势率，判断 3D-OCSB 方案前提是否成立。
不训练任何新模型。

## 结论

**前提成立，对应老师预案里的"情况 B"：可以推进，但 B 需要 Top-K 截断。**

| 判据（老师文档） | 目标 | 实测（τ=0.6，502 episodes） | 结论 |
|---|---|---|---|
| B 非空率 | > 80% | **100%** | ✅ |
| B 准确率 | > 60% | **39.1%** | ❌ 未达，但远超 30% 异常线 |
| 遗漏前景占比 | 30–60% | **30.3%** | ✅ 压在下沿 |
| 桥接优势率（单原型） | > 50% | **96.4%** | ✅ 远超 |
| 桥接优势率（多原型，COSeg 实际口径） | > 50% | **81.9%** | ✅ 远超 |

- **有矿但不多**：30.3% 的真实前景被初始模型漏掉（epoch=0 时是 47.9%，开 base guidance 后模型变准，遗漏变少）。
- **桥接非常有用**：对真遗漏前景，正确桥梁比 support 原型更接近的比例是 **96.4%（单原型）/ 81.9%（多原型）**。
- **B 仍不够纯**：39.1% 是真前景，其余是高置信误分。但假桥梁和真桥梁在"吸引遗漏前景"上**不可区分（95.8% vs 96.4%）**，说明 CSR 的负对照必须换一个更强的信号。

## 实验设置

| 项 | 值 |
|---|---|
| 模型 | COSeg 官方代码，`sc0_1w1s`（few-shot）+ `sc_s0pre`（backbone） |
| 关键参数 | `epoch=100`（启用 Base Prototypes Calibration，见下方"踩坑记录"） |
| 数据集 | ScanNet v2 val，312 scene |
| 类别划分 | CVFOLD 0，novel = bathtub/bed/bookshelf/cabinet/chair/counter/curtain/desk/door/floor |
| 超点 | 官方 `_vh_clean_2.0.010000.segs.json`（每点一个 segment id） |
| block 切分 | `room2blocks.py`，block_size=1, stride=1, min_npts=1000 |
| episode | 1-way 1-shot，502 有效 episodes（6 shard 并行，不同 seed） |
| S / Q | 同类、不同 block；同 scene 的两个 block 被禁止 |
| U | M=3 个 block，来自与 S/Q 不同的 scene，**且保证每个 block 都含目标类**（`u_with_class=3` 实测 100%） |
| 超点概率 | `final_pred`（插值回原始点）按 seg id 求均值 |
| 超点特征 | backbone 下采样特征，KDTree 映射回原点的 seg 后聚合 |
| 桥接优势 | `max Sim(B_pos, C_pos) > Sim(A_single, C_pos)`，A 为 support 前景特征均值 |

## 踩坑记录：epoch=0 关掉了 Base Guidance

COSeg 的 **Base Prototypes Calibration (BPC)** 在 `epoch < 1` 时是关闭的（训练第 0 epoch 的 warm-up，base 原型还没建立）。我们一开始传 `epoch=0`，让训练好的模型走了 warm-up 分支，BPC 被关掉。

实测代价：

| 配置 | query IoU（100 ep） | B 准确率 | 遗漏率 |
|---|---:|---:|---:|
| epoch=0（错） | 0.210 | 0.213 | 0.479 |
| epoch=100（对） | **0.378** | **0.391** | **0.303** |
| 论文（ScanNet 1-way 1-shot） | 0.420 | — | — |

差了 **16.8pp 的 query IoU**。这也解释了为什么第一版 B 特别脏：base 区域被大量误判成 novel，全混进了 B。

**这本身可以写进论文的 Ablation**：Base Guidance 是半监督超点桥接不可或缺的前置过滤器。

## Gate 3：block 与超点粒度

| 指标 | 值 |
|---|---|
| block 总数 | 7841 |
| 覆盖 scene | 312 / 312 |
| 每 block 点数 | 均值 5515，中位数 4752 |
| 每 block 超点数 | 均值 47.2，中位数 38 |
| 超点 < 5 的 block | 113（1.4%） |
| 每 block 类别数 | 均值 3.3 |

每类可用 block：floor 7162、chair 2188、cabinet 828、door 794、desk 638、
bed 553、counter 344、bookshelf 279、curtain 196、bathtub 110。

## B/C 总体统计（502 episodes）

| τ_b | B 非空率 | B 准确率 | soft precision | 遗漏占比 |
|---|---:|---:|---:|---:|
| 0.5 | 1.000 | 0.373 | 0.373 | 0.303 |
| 0.6 | 1.000 | 0.391 | 0.391 | 0.303 |
| 0.7 | 0.998 | 0.402 | 0.402 | 0.303 |
| 0.8 | 0.996 | 0.418 | 0.418 | 0.303 |

桥接优势率（n=408 有有效 C_pos 和 B_pos）：

| 原型口径 | 优势率 |
|---|---:|
| 单原型 A | **0.964** |
| 多原型 A（100 子原型取 max） | **0.819** |

**多原型的意义**：即使 support 用 100 个局部子原型覆盖类内变化，仍有 81.9% 的遗漏前景无法被任何子原型覆盖，必须靠桥梁 B 来补。

## 假桥梁检验（CSR 的必要性）

| 指标 | 值 |
|---|---:|
| 假桥梁对遗漏前景的致幻率（`Sim(B_neg,C_pos) > Sim(A,C_pos)`） | **0.958** |
| 真桥梁的桥接优势率 | 0.964 |
| 真桥梁 vs support 负类 D | 0.561 ± 0.138 |
| 假桥梁 vs support 负类 D | 0.576 ± 0.144 |
| per-episode `Bneg_D > Bpos_D` 比例 | 57.7% |
| Cohen's d | 0.170 |

**两个结论**：

1. **假桥梁在"吸引遗漏前景"上和真桥梁几乎一样**（95.8% vs 96.4%）——只用 `Sim(B,C)` 无法区分真假，CSR 不是可选项。
2. **support 负对照的区分度很弱**（d=0.17，57.7% 接近抛硬币）——`η_c(B)` 如果只用 support 负类，可能压不住假桥梁。需要换更强的信号（见"下一步"）。

## Per-class（τ_b = 0.6，502 episodes）

| class | 名称 | n | B 非空率 | B 准确率 | 遗漏占比 | 优势率 |
|---|---|---:|---:|---:|---:|---:|
| 2 | floor | 47 | 1.00 | 0.426 | 0.351 | 0.945 |
| 3 | chair | 56 | 1.00 | 0.531 | 0.243 | 0.957 |
| 5 | desk | 40 | 1.00 | 0.269 | 0.230 | 0.957 |
| 6 | bed | 55 | 1.00 | 0.335 | 0.323 | 0.965 |
| 7 | bookshelf | 47 | 1.00 | **0.565** | 0.244 | 0.976 |
| 10 | bathtub | 48 | 1.00 | 0.261 | **0.450** | 0.974 |
| 12 | curtain | 46 | 1.00 | **0.582** | 0.218 | 0.988 |
| 13 | counter | 48 | 1.00 | **0.207** | 0.332 | 0.920 |
| 14 | door | 59 | 1.00 | 0.429 | **0.210** | **0.995** |
| 19 | cabinet | 56 | 1.00 | 0.289 | **0.421** | 0.963 |

**读法**：
- **curtain / bookshelf / chair 最健康**：B 准确率 0.53–0.58，遗漏 0.22–0.24。
- **counter 最脏**：B 准确率 0.207，但遗漏只有 0.332——模型把 counter 高置信地认成了 table/cabinet，这些点进了 B 而不是 C。
- **bathtub / cabinet 遗漏最多**：0.45 / 0.42，是桥接最该发力的类。
- **优势率全线 > 0.92**，door 最高 0.995。

## 局限

1. **U 是 oracle 采样**：为了保证 U 含目标类，U 从"包含目标类的 block 池"里选，用了 GT。
   实测 100% 的 episode 的 3 个 U block 都含目标类。测试时不可用，需要换成无 GT 的采样或判别器。
2. **B 的 raw precision 39.1% 不是方法上限**：它只是"简历池"的纯度，CSR 的面试能筛掉多少还没测。
3. **support 负对照信号弱**（d=0.17）：CSR 的 η 需要换设计。
4. **评测口径**：query IoU 0.378 vs 论文 0.420，差异来自点级 IoU vs per-class mIoU、episode 数（100 vs 1000）、voxel_size 等。

## 下一步建议

1. **B 改成 `τ_b + Top-K` 截断**：每个 episode 只留置信度最高的 Top-5 个超点，预期能把 B 准确率推到 50%+。
2. **换 CSR 的负对照信号**：support 负类的区分度不够（d=0.17）。候选：
   - base 子空间正交性：B 的特征是否被任何 base 类原型解释
   - B 与 C 的 Δ 分布（真假桥梁的 Δ 大小可能有差异）
   - 跨 episode 的稳定性（同一 B 在多个 episode 中是否一致）
3. **补逐桥 AUC**：现在只记了 episode 平均的 `max_D`，需要逐桥分数才能算 B_pos vs B_neg 的 AUC。
4. **换无 GT 的 U 采样**：训练一个轻量的"U 是否含目标类"判别器，或改成"U 全部 block + 按类条件统计"。
5. **扩到 2-way 1-shot**：验证多类竞争下 B/C 的变化。

## 复现路径

```bash
# 环境
conda activate /data01/jh/conda_envs/coseg   # python3.7 + torch1.11+cu113
export PYTHONPATH=/data01/jh/tmp/py_extra:/data01/jh/code

# 数据
/data01/jh/data/scannet_raw/scans            # 原始 ScanNet v2
/data01/jh/data/ScanNet/scenes/data          # (N,8) XYZRGB+label+segid
/data01/jh/data/ScanNet/blocks_bs1_s1/data   # 7841 blocks

# 权重
/data01/jh/weights/sc0_1w1s.pth
/data01/jh/weights/sc_s0pre/model/model_best.pth

# 诊断（6 shard 并行，每 shard 不同 seed）
cd /data01/jh/code
CUDA_VISIBLE_DEVICES=<gpu> python -u diag_bc.py --episodes 84 --M 3 --seed <s> \
  --out /data01/jh/data/diag_v4_s<s>

# baseline sanity gate
CUDA_VISIBLE_DEVICES=<gpu> python -u eval_baseline.py
```

产物：
- `/data01/jh/data/diag_v4_s*/records.jsonl`（355 ep）
- `/data01/jh/data/diag_v5_s*/records.jsonl`（147 ep）
- 本机 `ps_diag/diag_bc.py`、`eval_baseline.py`、`collect_scannet_data_sp.py`、`analyze_diag.py`
