# 脚本索引

生成时间 2026-11-17。所有脚本原先平铺在仓库根目录与 `afford_scripts/`，
现按研究阶段归档。服务器上的运行路径仍是 `/data01/jh/code/<basename>.py`，
本地目录结构只用于阅读与版本管理。

| 目录 | 文件数 | 用途 | 状态 |
|---|---:|---|---|
| `ocsb_now/` | 17 | 当前主线（OCSB 第一/二/三步） | 在用；门控已证伪，固定融合 +0.67 |
| `lib/` | 7 | 被 import 的核心库 | 禁止移动/改名 |
| `tools/` | 16 | 数据准备 / 缓存构建 / 可视化 | 可重跑 |
| `_archive/a1_early_signal/` | 31 | A1 早期信号搜索（11/05–11/06） | 已归档 |
| `_archive/a2_ocsb_skeleton/` | 23 | A2 OCSB 忠实实现与 CMC 注入（11/12–11/13） | 已归档 |
| `_archive/a3_phases/` | 12 | A3 四阶段方法（11/13） | 已归档 |
| `_archive/a4_rings/` | 10 | A4 四环诊断（11/13–11/16） | 已归档 |
| `_archive/a5_weak_support/` | 1 | A5 弱标注支持集摸底（11/16） | 已归档 |
| `_archive/a6_afford_diag/` | 21 | A6 Affordance 线诊断（11/16–11/17） | 已归档 |
| `_archive/a7_env/` | 4 | A7 环境搭建（11/16） | 一次性 |

## `ocsb_now/` — 当前主线（OCSB 第一步/第二步）

**状态**：★ 在跑，不要动 ｜ **对应报告**：—（第一步结果待写入报告）

| 脚本 | 行 | 用途 |
|---|---:|---|
| `ocsb_core.py` | 241 | 共用底座：种子锁、模型加载、episode 构造、`feats_one`/`build_proto`/`predict_agg`/`build_u_sparse`/`iou_counts` |
| `upper_bound.py` | 151 | 主实验：Baseline + Naive-B/Oracle-B/Oracle-B+C × 6 档 λ，只剩流程编排 |
| `post_ub560.py` | 254 | 后处理：合并分片 + 配对 t + 2 折 CV + 逐类 + 难度分层；带跨轮 Baseline 校验 |
| `check.py` | 139 | 三种校验合一个入口：`replicate`（对比原生 forward）/ `same`（同进程）/ `cross`（跨进程） |
| `ocsb_advanced.py` | 355 | OASB + CSR + λ_adap 的五步实现；**负结果，保留作记录** |
| `purity_analysis.py` | 198 | 12 个提纯信号的 AUC 分析（B 在特征空间不可分） |
| `purity_fix_probe.py` | 200 | 验证 FPS 离群/超点采样/反向管道校验三个根因 |
| `cross_block_probe.py` | 205 | 跨块互验证（U1↔U2 / U↔Q / LOO 支持块） |
| `gate_probe.py` | 143 | 门控信号探针：找到 sim_fg_q（AUC 0.751） |
| `ocsb_v3.py` | 270 | 三层遗漏条件化桥接：P1 提纯 / P2 单向 / P3 门控 |
| `post_v3.py` | 199 | v3 后处理：2 折 CV 选参 + 分层 + TP/FP/FN |
| `v3_build_cache.py` | 175 | 缓存 BridgeScorer 的 10 维相关性特征 |
| `train_bridge_scorer.py` | 160 | 训练 10→16→1 打分器（193 参数，测试折 AUC 0.70） |
| `v3_build_gate_cache.py` | 200 | 旧版门控缓存（含 Bug A/B/C，仅留档） |
| `v3g_build_cache.py` | 230 | 修复版缓存：块择优 / B-C 分档 / solo+融合双路 / 折外 scorer |
| `train_omission_gate.py` | 200 | 逐点门控（泄漏版，结果作废） |
| `train_episode_gate.py` | 215 | 任务级门控 11 参数（干净两折，已验证归零） |
| `v3i_intra_query.py` | 185 | 查询集内自桥接（U=Q），含随机选点对照 |

## `lib/` — 被 import 的核心库

**状态**：禁止移动/改名 ｜ **对应报告**：ocsb_now 的四个脚本都依赖 diag_bc + ring4_isolated

| 脚本 | 行 | 用途 |
|---|---:|---|
| `build_ocsb_cache.py` | 273 | 生成 ub560 / val_560 特征缓存 |
| `clip_vit.py` | 125 | CLIP ViT-B/16 视觉塔（TorchScript 重建） |
| `diag_bc.py` | 1687 | 核心：build_model / 缓存 / 评估，1687 行 |
| `fake_tp.py` | 79 | torch_points_kernels → pointops2 shim |
| `ocsb_cmc.py` | 264 | ocsb_faithful + 三个结构性修正 |
| `ocsb_faithful.py` | 420 | 导师文档 Method V1 式(1)-(19) |
| `ring4_isolated.py` | 362 | 提供 prepare_block()，ocsb_now 依赖 |

## `tools/` — 数据准备 / 缓存构建 / 可视化

**状态**：可重跑 ｜ **对应报告**：只在需要重建数据时用

| 脚本 | 行 | 用途 |
|---|---:|---|
| `build_intra_cache.py` | 83 | 补齐 intra-query 方向需要的字段。 |
| `build_train2k.py` | 87 | 从 train split 的 blocks 采 2000 个 base 类任务并提取缓存。 |
| `check_coseg.py` | 121 | Minimal COSeg forward check: load weights, run one support+query episode. |
| `collect_scannet_data_sp.py` | 159 | Collect ScanNetV2 point clouds + semantic labels + superpoint ids. |
| `collect_train_blocks.py` | 59 | 并行收集 ScanNet train split 的场景 npy（第 8 列保留 seg id）。 |
| `diag_cache.py` | 99 | 检查缓存质量与 OCSB 中间量的分布。 |
| `eval_baseline.py` | 85 | Baseline query mIoU for COSeg 1-way 1-shot on ScanNet val. |
| `export_full_room.py` | 89 | 导出 3 个典型任务对应的完整 ScanNet 房间（真彩 + 切块高亮）。 |
| `export_ply2.py` | 155 | 导出改进版 PLY： |
| `extract_orth_feats.py` | 157 | Phase 1: build an orthogonal visual feature f_orth with a frozen 2D encoder. |
| `make_manifest.py` | 160 | Build a fixed paired episode manifest. |
| `probe_repro.py` | 63 | Minimal reproducibility probe: same manifest episode, same GPU, two runs. |
| `rebuild_meta.py` | 16 | 从 manifest + 已存在的 npz 重建 val_meta（不依赖分段进程写文件）。 |
| `render_ply.py` | 102 | 把导出的 PLY 渲染成 PNG（3 视角，区域点高亮、背景淡化）。 |
| `scan_block_classes.py` | 17 | 一次性扫描所有 block 的类别，缓存为 json。 |
| `stats_blocks.py` | 34 | Gate-3 check: block count and superpoints per block. |

## `_archive/a1_early_signal/` — A1 早期信号搜索（11/05–11/06）

**状态**：已归档 ｜ **对应报告**：STEP_OCSB_EXPLORATION / STEP_OCSB_RESULT / STEP_OVERNIGHT_REPORT / REPORT_TO_ADVISOR

| 脚本 | 行 | 用途 |
|---|---:|---|
| `adaptive_signal.py` | 139 | Task 2e: per-class adaptive signal selection. |
| `analyze_cal.py` | 87 | Task 10: does margin-space calibration / a threshold-free soft gate beat the |
| `analyze_diag.py` | 48 | Summarize diag_bc.py records: overall + per-class tables. |
| `analyze_full.py` | 85 | Task 2/3 on the full pool: old 502 (v4/v5) + new full-field run. |
| `analyze_overnight.py` | 146 | Overnight report analysis: Task 1/2/3/4 tables. |
| `analyze_paired.py` | 112 | Paired analysis over the fixed manifest. |
| `analyze_task6.py` | 177 | Task 6 analysis: drop-off anatomy, asymmetric fusion, RSV gating, K sweep. |
| `analyze_toplk_oracle.py` | 72 | Top-K bridge truncation + Oracle injection analysis. |
| `auc_signals.py` | 54 | AUC of candidate negative-control signals: B_pos vs B_neg per episode. |
| `cohens_topk.py` | 70 | Task 2 completion: Cohen's d + per-signal Top-K B precision. |
| `combo_signals.py` | 134 | Task 2c: linear combination search over orthogonal signals. |
| `compare_u_mode.py` | 139 | Compare cross-scene vs same-scene U without relying on episode pairing. |
| `control_u_confound.py` | 121 | Control for the difficulty confound in the cross/same-scene comparison. |
| `local_geometry.py` | 127 | Task 2g: superpoint local geometry (curvature / planarity) as a signal. |
| `perclass_overnight.py` | 100 | Per-class breakdown of the overnight experiments (offline). |
| `phase2_clip.py` | 119 | Phase 2: does the CLIP referee break the 0.610 AUC ceiling? |
| `phase2_orth.py` | 107 | Phase 2: does the physical-appearance referee separate B_pos from B_neg |
| `phase3_gate.py` | 67 | Phase 3: does the CLIP consensus gate predict when bridging helps? |
| `phase4_veto.py` | 100 | Task 14: orthogonal veto as the episode gate, after fixing the eta scale. |
| `query_gate.py` | 67 | Task 1.5 completion: real query-correlation gate for the bridge injection. |
| `run_k_sweep.sh` | 18 | Same-scene U with a wider end-to-end K sweep (3/5/10/20 x lam 0.3/0.5). |
| `run_paired.sh` | 25 | Step 1+2: paired manifest + balanced prototype quota. |
| `run_same_scene.sh` | 21 | same_scene U control run. Seeds match diag_gate_s12{1,2,3} so that |
| `run_task6.sh` | 21 | Task 6: asymmetric fusion + RSV gate + full K sweep, on both U protocols. |
| `same_scene_distance.py` | 110 | Does the same-scene gain decay with distance between query and U blocks? |
| `stability_signal.py` | 117 | Task 2f: cross-episode B stability as an orthogonal signal. |
| `step1_gate.py` | 115 | Step 1: add the end-to-end column to the baseline-stratified table, plus |
| `step23_gate.py` | 121 | Step 2/3: can an unsupervised episode-level signal predict (a) the baseline |
| `topk_curve.py` | 72 | How does B precision decay as K grows, for the scores we could use? |
| `u_block_diag.py` | 168 | Block/scene-level diagnostics for the bridge set B. |
| `watch_s2.sh` | 20 | Watchdog for the Stage II run: appends the latest validation point every |

## `_archive/a2_ocsb_skeleton/` — A2 OCSB 忠实实现与 CMC 注入（11/12–11/13）

**状态**：已归档 ｜ **对应报告**：FAITHFUL_REPORT / ZERO_LEAK_REPORT / PPT_CUMUL_REPORT / FINAL_VERDICT

| 脚本 | 行 | 用途 |
|---|---:|---|
| `death_test.py` | 216 | 生死线：Top-K% 桥梁精度能否超 I0 |
| `diag_cmc.py` | 71 | 诊断 delta_corr 是否对准真实漏检点。 |
| `diag_signals.py` | 113 | 四种候选信号在 TP/FN/FP/TN 上的四分组诊断。 |
| `diag_step2.py` | 85 | 诊断式(15) 融合后掉点的原因。 |
| `or_gate.py` | 163 | 实验 2：或门直接恢复（OR-Gate Recovery）。 |
| `or_gate2.py` | 116 | or_gate v2：B_max 与 oracle 在 val_560 上的 Arith + Cumul。 |
| `run_ablation.py` | 398 | 导师文档第五部分的五步消融。 |
| `run_cmc.py` | 174 | 在 val_100 上评测 CMC 注入（Fix 1 + Fix 2 + Fix 3）。 |
| `run_cmc_s2.py` | 113 | S2（纯加权相似度，中心化）的 CMC 注入评测。 |
| `run_s2.py` | 124 | S2（纯加权相似度）的端到端评测。 |
| `run_s34.py` | 106 | S3 / S4（减背景得分）的 Logit 注入评测。 |
| `selfcheck_4_4.py` | 150 | 导师文档 4.4 节的五项强制检查。 |
| `summarize_full.py` | 79 | 汇总全量 560 eps 的 Step 1-4 与 Step 5 网格结果。 |
| `sweep_fusion.py` | 103 | 扫描式(15)修正后的 fusion_mode 与 gamma。 |
| `sweep_gamma.py` | 75 | 扫 gamma：融合强度从 0 到 0.2，找不伤 IoU 的临界点。 |
| `sweep_gamma_fine.py` | 83 | Step 4 的 gamma 细化扫描（1.0-4.0），找最优点。 |
| `sweep_step2.py` | 109 | Step 2 掉点诊断：扫描 tau 与 Top-K，确认式(15)的量纲问题。 |
| `test_ocsb_unit.py` | 91 | ocsb_faithful 的纯 CPU 单元测试（不加载 COSeg，秒级）。 |
| `train_stage2.py` | 361 | Stage II: light fine-tuning of the bridge-path HCA only. |
| `train_sweep.py` | 100 | Step 5 的 W_phi 训练网格：w_pos / w_reg / lr，逐轮评测。 |
| `train_sweep2k.py` | 160 | 在 2000 个零泄漏 train split 任务上训练 W_phi，每 500 步评测 val_560。 |
| `verify_controls.py` | 185 | 三个对照，检验 Step 4 的增益是否来自真实空间匹配。 |
| `verify_gain.py` | 130 | 核查 +2.13pp 的两个乐观来源。 |

## `_archive/a3_phases/` — A3 四阶段方法（11/13）

**状态**：已归档 ｜ **对应报告**：INTRA_QUERY_REPORT

| 脚本 | 行 | 用途 |
|---|---:|---|
| `phase1_smooth.py` | 220 | Phase 1：查询集超点内部一致性平滑 + 2 折交叉验证。 |
| `phase2_adjacent.py` | 183 | Phase 2：查询集内部三维邻接约束下的自桥接 A -> B_Q -> C_Q。 |
| `phase2_full.py` | 285 | Phase 2 完整版：三种匹配度 + CSR 净贡献验证 + 2 折 CV。 |
| `phase2_v2.py` | 327 | Phase 2 完整版 v2：四种匹配度 + CSR 验证 + 2 折 CV + Cumul。 |
| `phase3_calib.py` | 116 | Phase 3：自适应防爆阈值 + 支持集去污染诊断。 |
| `phase3_erode3d.py` | 155 | Phase 3 补充：基于支持集三维距离 d_to_bg 的边界腐蚀。 |
| `phase3_full.py` | 169 | Phase 3 完整版：β_orth 正交投影 + 边界腐蚀 + 自适应阈值。 |
| `phase3_v2.py` | 214 | Phase 3 v2：β_orth / 边界腐蚀 / 自适应阈值，全部带 Cumul 与 2 折 CV。 |
| `phase4_p2_perclass.py` | 90 | 补齐 Phase 4 后两行的逐类对比：Phase 2 CV 盲测 + 去污染（同上一行）。 |
| `phase4_report.py` | 106 | Phase 4：完整消融表（独立行 + pSE + 逐类）。 |
| `phase4_v2.py` | 178 | Phase 4 v2：按 2 折 CV 盲测口径生成消融表。 |
| `phase4_v3.py` | 200 | Phase 4 v3：统一口径的 2 折 CV 盲测消融表。 |

## `_archive/a4_rings/` — A4 四环诊断（11/13–11/16）

**状态**：已归档 ｜ **对应报告**：RING12_REPORT / RING23_REPORT / RING4_REPORT

| 脚本 | 行 | 用途 |
|---|---:|---|
| `ring1_balance.py` | 106 | 第一环：数据与任务划分均衡性核查（10 个 novel 类）。 |
| `ring2_check.py` | 129 | 核查 ring2 几何统计的 4 处疑点。 |
| `ring2_export_ply.py` | 94 | 第二环：导出 8 个典型任务的 U 点云（原始 RGB + 区域标注）。 |
| `ring2_physical.py` | 200 | 第二环：B_pos / B_neg / C_pos 的语义成分与稠密几何特征。 |
| `ring2b_fixed.py` | 114 | 10 类 unannotated 统计 + B_pos/B_neg 物理离地高度（按 block 分别算地面）。 |
| `ring2b_unannotated.py` | 100 | 10 类 unannotated 统计 + B_pos/B_neg 物理离地高度分布。 |
| `ring2c_full.py` | 130 | 第二环补全：全 10 类 × 30 eps，按 block 自己的 z_min 算离地高度。 |
| `ring3_purify.py` | 171 | 第三环：B 提纯策略 A/B/C（全 10 类 × 30 eps）。 |
| `ring4_report.py` | 119 | 第四环出表：读 ring4_full.json，输出主表、纯度分组、Tier3 归因。 |
| `ring4_ucover.py` | 90 | 第四环补充：U 中真值同类物体对 Q 前景的形态覆盖。 |

## `_archive/a5_weak_support/` — A5 弱标注支持集摸底（11/16）

**状态**：已归档 ｜ **对应报告**：WEAK_SUPPORT_REPORT

| 脚本 | 行 | 用途 |
|---|---:|---|
| `weak_support_test.py` | 305 | 弱标注 S→Q 摸底，不引入 U |

## `_archive/a6_afford_diag/` — A6 Affordance 线诊断（11/16–11/17）

**状态**：已归档 ｜ **对应报告**：AFFORD_BASELINE_DIAG / AFFORD_CEILING_3DIR

| 脚本 | 行 | 用途 |
|---|---:|---|
| `afford_error_diag.py` | 191 | Affordance 基线四维错题诊断（只读 NAS，输出到 /data01/jh/afford_diag/）。 |
| `afford_oracle.py` | 200 | Oracle 拆解：无效配对置零 / 有效对内阈值校准，能带来多少指标提升。 |
| `afford_saturation.py` | 86 | 3D_AffordanceNet：APGraph 输出的饱和程度（逐对 p90/p95/p99 与 GT 占比）。 |
| `afford_summary.py` | 98 | 汇总：Base/Novel 拆分 + 逐类/逐动作 mAP + 误差性质。 |
| `apgraph_gate_probe.py` | 165 | APGraph 3DIR：拆开"历史激活校准门"对新动作的影响。 |
| `apgraph_saturation_probe.py` | 97 | Knife-stab 饱和归因：是 3D 特征本身不可分，还是属性映射溢出。 |
| `apgraph_scale_probe.py` | 177 | APGraph 3DIR：输出尺度校准 + Base 响应扩散检验。 |
| `compare_full_partial.py` | 93 | full vs partial 的逐类/逐动作对比（用 TP/FP/FN 算 IoU 与召回）。 |
| `dump_worst_ply.py` | 89 | 把 worst_samples.npz 导出为 ply（GT 绿 / 预测红 / 重叠黄 / 其余灰）。 |
| `iagnet_frozen_probe.py` | 92 | IAGNet 冻结双编码器的可塑性下限测试。 |
| `joint_compare.py` | 79 | 汇总：11 类 base / APGraph 增量后 / 23 类全监督 在三组口径上的对照。 |
| `joint_eval.py` | 130 | 评估全监督（23 类联合训练）DGCNN base：与 APGraph 的 38.60 / 33.76 同口径。 |
| `make_all23_manifest.py` | 120 | 构造 23 类联合（全监督）训练的 manifest。 |
| `make_all23_manifest_partial.py` | 134 | 构造 partial（单视角）协议的 23 类联合训练 manifest。 |
| `merge_ub.py` | 45 | 只合并汇总项、无法精确合并，已被 post_ub560 取代 |
| `perpair_breakdown.py` | 93 | 从 ljy 已有的 per_pair.csv 做逐物体类 / 逐动作拆解（只读，不跑模型）。 |
| `threedir_breakdown.py` | 274 | 3DIR / PIAD 误差解剖：逐物体类、逐动作、排序能力 vs 定位精度。 |
| `threedir_image_confound.py` | 90 | 3DIR：功能框大小 / 图像可用性 与逐对指标的关联。 |
| `threedir_point_dissect.py` | 84 | 逐点解剖：GT 占比、预测概率分位数、20 档阈值 IoU 曲线。 |
| `threedir_stage_trace.py` | 124 | 握持配对在 stage1→stage4 的轨迹：是"学不会"还是"学了又忘"。 |
| `upper_bound_test.py` | 149 | v1（有穿模/砍半/C_pos 三 bug），已被 v2 取代 |

## `_archive/a7_env/` — A7 环境搭建（11/16）

**状态**：一次性 ｜ **对应报告**：—

| 脚本 | 行 | 用途 |
|---|---:|---|
| `dl_ckpt.sh` | 19 | 下载 Sonata 主权重（hf-mirror 不支持 Range，只能整份重下） |
| `dl_torch.sh` | 24 | 16 线程分段下载 torch 2.5.0 whl（清华镜像单连接限速 ~204KB/s） |
| `install_deps.sh` | 31 | Sonata 环境依赖安装（torch 用本地 wheel，其余走清华镜像） |
| `sonata_check.py` | 134 | Sonata 环境验证（headless，不依赖 open3d 可视化）。 |

## 报告文件（仓库根目录）

| 报告 | 阶段 |
|---|---|
| `3D OCSB原始idea.md` | 最初 idea |
| `STEP_OCSB_EXPLORATION.md` | A1 探索 |
| `STEP_OCSB_RESULT.md` | A1 结果 |
| `STEP_OVERNIGHT_REPORT.md` | A1 通宵实验 |
| `REPORT_TO_ADVISOR.md` | A1 阶段汇报 |
| `FAITHFUL_REPORT.md` | A2 忠实实现 |
| `ZERO_LEAK_REPORT.md` | A2 零泄漏 |
| `PPT_CUMUL_REPORT.md` | A2 Cumul 口径 |
| `FINAL_VERDICT.md` | A2 结论 |
| `INTRA_QUERY_REPORT.md` | A3 phase1-4 |
| `RING12_REPORT.md` | A4 ring1-2 |
| `RING23_REPORT.md` | A4 ring2-3 |
| `RING4_REPORT.md` | A4 ring4 |
| `WEAK_SUPPORT_REPORT.md` | A5 弱标注 |
| `AFFORD_BASELINE_DIAG.md` | A6 affordance 基线 |
| `AFFORD_CEILING_3DIR.md` | A6 3DIR 上限（795 行） |
| `OCSB_OASB_CSR_REPORT.md` | OASB/CSR 重启验证：零训练提纯的否证 |
| `OCSB_V3_REPORT.md` | 三层遗漏条件化桥接（零训练基线 +0.71pp） |
| `OCSB_STAGE2_REPORT.md` | Stage II：门控探索与归零（固定融合 +0.67） |
| `OCSB_INTRA_QUERY_REPORT.md` | 查询集内自桥接：Arith +0.93（CI 显著） |

## 产物目录（非脚本，未移动）

| 目录 | 内容 |
|---|---|
| `results/` | ring2c / ring4 / ring4_ucover 的汇总 json |
| `ply/`, `ply_png/` | A4 ring2 导出的点云与渲染图 |
| `ply2/`, `ply2_png/` | A4 ring2 改进版导出与渲染图 |
| `afford_diag_ply/` | A6 affordance 错例点云 |
