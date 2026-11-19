# 3D-OCSB 项目（已结案）

最后更新：2026-11-19

> 这里其实是**两个独立 idea**，先后在同一个 codebase 上验证，都已结案。
> 它们**不是**同一个想法的两次尝试：研究对象、干预点、失败原因都不同，只有两个子模块被继承复用。

---

## Idea A：3D-OCSB（遗漏条件化选择性语义桥接）

| | |
|---|---|
| **问题** | 半监督小样本 3D 分割里，初始模型判为背景的区域中，有一部分是真正被漏掉的前景（遮挡/截断/异形）。支持集 A 解释不了它们，但某个"可信桥梁" B 可能可以。 |
| **机制** | 三类区域 A（支持锚点）/ B（候选桥梁）/ C（遗漏候选）；<br>OASB：比较 `A→C` 与 `A+B→C` 的预测变化；<br>CSR：用支持负对照判断 B 的贡献是否真有类别选择性，只恢复通过的 C。 |
| **载体** | COSeg 1w1s + 跨场景无标签池 U（560 个 novel episode） |
| **结论** | **证伪**。全量盲测 +0.26 / +0.05pp；并查出基线本身过预测前景，任何常数负偏移都能白捡 +1.3pp。 |
| **文档** | [`docs/ocsb_legacy/FINAL_VERDICT.md`](docs/ocsb_legacy/FINAL_VERDICT.md)（判决书），其余 21 份是同期的探索记录 |

## Idea B：GFS-VL 伪标签可靠性改造

| | |
|---|---|
| **问题** | GFS-VL 每个 iter 用 VLM 在线生成 PS/AI 伪标签，45M 点里精度参差（PS 0.574 / AI 新增 0.226）。能否先离线复现这些伪标签、再用独立的几何视角给每个点打可靠性分，以此加权训练？ |
| **机制** | 离线复现 45M 点级伪标签 → 逐点 Sonata 几何一致性打分（AUC 0.783）→ 软权重 CE；<br>另外把 Idea A 的「桥梁池」和「基类负对照」搬过来复测。 |
| **载体** | ScanNet200 GFS（57 类 = 12 base + 45 novel，1-shot，312 val），GFS-VL 官方权重 + 自训同源基线。 |
| **与 A 的关系** | **只继承了两个子模块**（桥梁池 B、基类负对照 D），二者在本载体上重新测量并再次失败（0.496 / 0.754）；<br>B 的核心问题——"已生成伪标签的可靠性校准"——在 A 里并不存在。 |
| **结论** | **结案**。方法效应（−0.4 ~ +0.7pp，符号翻转）小于 seed 噪声（0.9–1.9pp）；分类头偏置上限 0pp；瓶颈在 1-shot 新类分类器，不在伪标签。 |
| **文档** | **[`GFS_VL_CONCLUSION.md`](GFS_VL_CONCLUSION.md)** |

## 两个 idea 的边界

| | Idea A | Idea B |
|---|---|---|
| 研究对象 | 背景预测区里的**遗漏前景** | 已生成伪标签的**逐点可靠性** |
| 干预时机 | 推理期恢复（零训练） | 训练期损失加权 |
| 关键失败证据 | 桥梁 AUC 0.496；负对照 0.754 < 纯原型 0.795 | 软权重跨 seed 符号翻转；δ 扫描上限 0pp |
| 共同结论 | 1-shot 3D 的瓶颈在**新类表征**，不在"怎么补/怎么加权" | 同左 |

---

## 目录

```
.
├── GFS_VL_CONCLUSION.md      ← Idea B 的完整结论与探索过程（含全部 log 路径）
├── 3D OCSB原始idea.md         ← Idea A 的原始 Method 初稿（顶部已加结案指针）
├── docs/ocsb_legacy/          ← Idea A 时期的 22 份报告
├── docs/SERVER_CACHE_INDEX.md ← **服务器缓存索引**（哪些缓存在、能干什么、重建多久）
├── ocsb_now/                  ← 两个 idea 的脚本（GFS 阶段的对应服务器 /data01/jh/code/ps_diag）
├── weights/                   ← 本地权重备份（sonata.pth，414MB）
├── wheels/                    ← 离线安装包（spconv / torch_scatter，81MB）
├── _archive/                  ← a1~a7 旧阶段；a8_viz 可视化（~130MB）；a9_misc 旧 json
└── .git
```

## 服务器端关键资产（/data01/jh，未整理）

| 内容 | 路径 |
|---|---|
| 点级伪标签（在线 PS / PS+AI） | `/data01/jh/data/online_pl/{scene}.npz`（37MB） |
| 软权重 | `/data01/jh/data/online_pl_w/{scene}_{pl,w}.npy`（1003MB） |
| 逐点 Sonata 分数 | `/data01/jh/data/pp_score/{scene}_s.npy`（336MB） |
| 区域缓存 / Sonata 区域特征 | `/data01/jh/data/region_cache`、`sonata_region`、`sonata_base` |
| 实验权重 | `/data01/jh/code/GFS-VL/exp/scannet200/sc200_k1_*` |
| 全部日志 | `/data01/jh/afford_diag/gfs_build/*.log` |
| ScanNet++ 转存（网盘） | `/scannetpp/<scene>/scans`，954 场景，~142GB |

## 如果以后想接着做

- 唯一被实测证实的杠杆是**训练量**：20→40 epoch 让 Novel 从 0.2491 涨到 0.2842（+3.5pp）；
- 或者换战场：新类分类器 / 原型质量，而不是伪标签或遗漏恢复。
