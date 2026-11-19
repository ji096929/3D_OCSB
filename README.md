# 3D-OCSB 项目（已结案）

最后更新：2026-11-19

## 当前状态：两条路线都已证伪，方向停在这里

| 路线 | 载体 | 结论 | 证据 |
|---|---|---|---|
| 路线 1：跨场景桥接（外部无标签池 U） | COSeg 1w1s | **证伪**：560 episode 盲测 +0.26 / +0.05pp；基线本身过预测，常数负偏移能白捡 +1.3pp | [`docs/ocsb_legacy/FINAL_VERDICT.md`](docs/ocsb_legacy/FINAL_VERDICT.md) |
| 路线 2：GFS-VL 伪标签改造 | ScanNet200 GFS | **结案**：提纯/补标/负对照/软权重全部无超噪声增益；瓶颈在新类分类器，不在伪标签 | **[`GFS_VL_CONCLUSION.md`](GFS_VL_CONCLUSION.md)** |

**先读 `GFS_VL_CONCLUSION.md`**：里面有完整实验矩阵、每个机制的具体数字、失败原因分析、以及可复用资产（离线伪标签管线、权重通路、训练守护、诊断脚本）。

## 目录

```
.
├── GFS_VL_CONCLUSION.md      ← 本轮（GFS-VL）的完整结论与探索过程
├── 3D OCSB原始idea.md         ← 原始 idea（A/B/C 三类区域 + OASB + CSR）
├── docs/ocsb_legacy/          ← 路线 1（COSeg）时期的 22 份报告（FINAL_VERDICT / RING / OCSB_* 等）
├── ocsb_now/                  ← 路线 2 的脚本（与服务器 /data01/jh/code/ps_diag 对应）
├── weights/                   ← 本地模型权重备份（sonata.pth，414MB）
├── wheels/                    ← 离线安装包（spconv / torch_scatter，81MB，网络差时有用）
├── _archive/                  ← a1~a7 旧阶段产物；a8_viz 可视化（ply/png，~130MB）；a9_misc 旧 json
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

## 一句话总结为什么停

1. 同方法不同 seed 的 Novel 波动 **0.9–1.9pp**，而所有伪标签改动的方法效应在 **−0.4 ~ +0.7pp** 且符号翻转；
2. 分类头「基类→新类」偏置扫描的最优点就是 δ=0，**上限 0pp**；
3. 新类误差均匀分布在「错成基类」和「新类互混」，没有单点主矛盾。

## 如果以后想接着做

- 唯一被实测证实的杠杆是**训练量**：20→40 epoch 让 Novel 从 0.2491 涨到 0.2842（+3.5pp）；
- 或者换战场：新类分类器/原型质量，而不是伪标签。
