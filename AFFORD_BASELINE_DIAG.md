# Affordance 基线细粒度诊断

日期：2026-11-16
约束：`/nas_data/ljy/` 全程只读；所有输出在 `/data01/jh/afford_diag/`。
环境：`/data01/jh/conda_envs/sonata`（Python 3.10 + torch 2.5.0+cu124，补装了 cupti/nccl 与 9 个 Python 包）。

---

## 0. 结论

1. **误差主体不是"漏检"，是"过度激活"。** 有效配对内，新类召回只有 **0.426**，而假阳性点数是真实正例的 **1.92 倍**；旧类召回 0.768、FP/pos 1.75。预测出来的区域平均是真实区域的 3 倍大。
2. **85.6% 的假阳性来自"物体根本不支持该动作"的组合。** 无效 (object, affordance) 对贡献 8,655,199 个误报点，有效对内部只有 1,458,719。模型完全没有"物体—动作有效性"的概念，只学了"这块局部形状像不像某动作"。
3. **新类的真正短板是召回**（0.426 vs 旧类 0.768），而误报率两类几乎一样。这说明新类卡在"找不到"，不是"乱标"。
4. **从 1-shot 到 5-shot，新类 mAP 只涨 1.94pp**（31.44 → 33.38）。增加支持样本的边际收益极低，与之前弱标注实验的结论一致：**支持集这一侧不是瓶颈**。
5. **单视角残缺（partial）主要损失召回**：新类 0.426 → 0.337（−8.9pp），旧类 0.768 → 0.683（−8.5pp），而溢出几乎不变（1.918 → 1.902）。
6. **GREAT / GEAL 在这个方向上并没有被解决。** 在 3DIR 协议下 5-shot Novel 分别只有 9.13 / 13.79，远低于 APGraph 的 16.54，且 Base 遗忘高达 35–36pp。

---

## 1. 基线核查：两套协议，别混

| | **3D_AffordanceNet** | **3DIR / PIAD** |
|---|---|---|
| 物体类 | 23 类 = 11 base + 4+4+4 novel（**12 新类**） | 同左（3DIR 重排阶段顺序） |
| 动作 | **18** | **17**（3DIR 上 8 novel） |
| shot | 1 / 2 / 5 | **3 / 5** |
| 已有对比方法 | **只有 FT、START**（其余全空） | **10 个方法完整** |

**3DIR 协议 5-shot 主表（`baseline/docs/main_results_3dir.md`）**：

| 方法 | Base aIOU | All | Novel | AUC | Base 遗忘 |
|---|---:|---:|---:|---:|---:|
| IAGNet | 40.76 | 9.51 | 14.74 | 57.30 | 35.90 |
| **GREAT** (CVPR 2025) | 40.00 | 6.50 | **9.13** | 58.22 | 35.84 |
| MIFAG | 39.05 | 7.19 | 9.81 | 57.53 | 34.19 |
| **GEAL** (CVPR 2025) | 40.25 | 9.76 | **13.79** | 61.35 | 34.06 |
| SequentialFT | 43.37 | 6.59 | 9.94 | 70.69 | 39.76 |
| FrozenHead | 43.37 | 10.25 | 11.31 | 57.77 | 34.06 |
| LwF | 43.37 | 7.68 | 12.02 | 70.46 | 39.56 |
| EWC | 43.37 | 21.58 | 16.96 | 79.82 | 17.68 |
| **APGraph**（ljy） | 43.37 | **30.72** | **16.54** | **84.16** | **0.06** |

**GREAT / GEAL 解决的问题是"语言/视觉引导"和"通用增量"，不是这里的痛点。** 它们的 Novel aIOU 只有 9–14，和 APGraph 的 16.54 差 3–7pp，更关键的是 Base 遗忘 34–36pp——增量后旧类基本崩了。

另外：`GREAT`、`MIFAG`、`START_PointNet2` 在 `baseline/` 下**没有可直接读取的评估 json**（只有 protocol 计数和 smoke 结果），3D_AffordanceNet 协议的主表里它们的行是空的。真正有完整结果的是 3DIR/PIAD 那套。

---

## 2. Base / Novel 拆分

### 2.1 划分（`manifests/k5_seed1.json`）

| Session | 类别 |
|---|---|
| 0（base，11） | Bag, Bed, Bowl, Chair, Clock, Door, Earphone, Hat, Knife, Laptop, Refrigerator |
| 1（novel，4） | Dishwasher, Faucet, Scissors, Vase |
| 2（novel，4） | Microwave, Mug, StorageFurniture, Table |
| 3（novel，4） | Bottle, Display, Keyboard, TrashCan |

共享动作（11 个）：`grasp, contain, openable, support, wrap_grasp, pourable, move, displaY, press, cut, stab`
阶段专属动作（7 个）：`lift, layable, sittable, pushable, pull, listen, wear`

### 2.2 分组 mAP（DGCNN，5 seeds 平均，`per_pair.csv` 口径）

| shot | Base mAP | Novel mAP | All mAP | Base AUC | Novel AUC |
|---|---:|---:|---:|---:|---:|
| 1 | 43.64 | 31.44 | 37.33 | 85.47 | 76.16 |
| 2 | 43.77 | 32.07 | 37.72 | 85.55 | 76.68 |
| 5 | 43.74 | 33.38 | 38.38 | 85.41 | 78.58 |

**注意口径**：`eval_k5_seed1/metrics.json` 的 `mAP = 40.96` 是 `--all-objects`（全部 2285 物体、58 个有效对）的 seed1 单次结果；上面的 38.38 是 5 seeds × per-pair 宏平均。两者不是同一个数。

### 2.3 12 个 Novel 类逐类 mAP（DGCNN）

| 类别 | k1 | k2 | k5 | k5−k1 |
|---|---:|---:|---:|---:|
| Display | 69.21 | 69.02 | 68.81 | −0.40 |
| Dishwasher | 47.54 | 47.41 | 47.12 | −0.42 |
| Scissors | 43.77 | 42.96 | 46.01 | +2.25 |
| TrashCan | 37.68 | 38.43 | 39.86 | +2.18 |
| Bottle | 31.41 | 32.97 | 35.69 | +4.28 |
| StorageFurniture | 32.48 | 32.72 | 32.13 | −0.34 |
| Microwave | 33.16 | 33.45 | 33.39 | +0.23 |
| Mug | 27.88 | 28.67 | 30.11 | +2.23 |
| Vase | 23.08 | 24.57 | 25.02 | +1.94 |
| Table | 19.51 | 19.65 | 20.30 | +0.79 |
| **Faucet** | 7.92 | 9.00 | **12.17** | +4.25 |
| **Keyboard** | 10.62 | 11.10 | **10.84** | +0.21 |

**novel 内部差异极大**：Display 68.81 vs Keyboard 10.84，差 6 倍。而且 **3 个类的 5-shot 反而不如 1-shot**（Display/Dishwasher/StorageFurniture），说明多支持样本在这个协议下并不是单调增益。

### 2.4 18 个动作逐动作 mAP

| 动作 | k1 | k5 | k5−k1 |
|---|---:|---:|---:|
| stab | 80.84 | 80.26 | −0.58 |
| lift | 70.66 | 70.66 | 0.00 |
| displaY | 58.20 | 58.04 | −0.16 |
| layable | 53.17 | 53.17 | 0.00 |
| contain | 52.00 | 53.11 | +1.11 |
| sittable | 51.63 | 51.63 | 0.00 |
| pull | 49.71 | 49.71 | 0.00 |
| support | 36.63 | 36.93 | +0.31 |
| cut | 36.43 | 40.10 | +3.67 |
| listen | 35.27 | 35.27 | 0.00 |
| pourable | 33.84 | 35.34 | +1.51 |
| grasp | 29.30 | 31.75 | +2.45 |
| openable | 26.17 | 27.76 | +1.59 |
| press | 25.60 | 25.73 | +0.13 |
| move | 24.32 | 24.35 | +0.04 |
| pushable | 19.91 | 19.91 | 0.00 |
| **wear** | 16.82 | 16.82 | 0.00 |
| **wrap_grasp** | **9.60** | **9.96** | +0.35 |

7 个动作的 k1/k2/k5 完全相同（`lift/layable/sittable/pushable/pull/listen/wear`）——它们是**阶段专属动作**，只在 base 类出现，而 base 原型在增量前就冻结了。

**`wrap_grasp`（9.96）和 `wear`（16.82）是全表最差。** 这两个动作都需要理解"凹形包围关系"，而不是局部凸起；最好的 `stab`（80.26）和 `lift`（70.66）恰恰都是"局部尖端点/顶面"。

---

## 3. 误差性质拆解（5-shot seed1，逐点阈值 0.5）

### 3.1 三种错误的总量

| 分组 | 有效对 | TP | FP | FN | 召回 | FP/pos |
|---|---:|---:|---:|---:|---:|---:|
| novel | 30 | 181,811 | 818,044 | 244,802 | **0.426** | **1.918** |
| base | 28 | 280,510 | 640,675 | 84,655 | **0.768** | 1.754 |

**溢出（FP/pos）两类都接近 1.8–1.9**，也就是说模型标出的前景面积平均是真实区域的 **2.9 倍**。

### 3.2 无效物体-动作误报（最大的一类）

| 项 | 点数 | 占全部 FP |
|---|---:|---:|
| 无效 (object, affordance) 对的误报 | **8,655,199** | **85.6%** |
| 有效对内部的 FP | 1,458,719 | 14.4% |

Top 10（全部是 Chair / Table）：

| 物体 | 动作 | 分组 | 误报点 |
|---|---|---|---:|
| Chair | cut | base | 410,520 |
| Chair | lift | base | 333,669 |
| Table | layable | novel | 318,888 |
| Chair | grasp | base | 318,671 |
| Chair | wear | base | 259,168 |
| Chair | listen | base | 232,346 |
| Table | wear | novel | 230,933 |
| Chair | layable | base | 230,307 |
| Chair | press | base | 225,042 |
| Table | listen | novel | 207,479 |

Chair 和 Table 是误报大户（它们本来是点数/实例最多的类：9035 / 7845 实例）。**模型对"椅子不能 cut"这件事完全没有概念。**

### 3.3 有效配对内：召回 vs 溢出

召回最低的 (class, aff)：

| 物体 | 动作 | 召回 | 精度 |
|---|---|---:|---:|
| Bottle | wrap_grasp | **0.039** | 0.068 |
| Faucet | openable | 0.093 | 0.030 |
| Bottle | grasp | 0.150 | 0.236 |
| Table | move | 0.246 | 0.119 |
| Vase | wrap_grasp | 0.267 | 0.131 |

溢出最高的 (class, aff)：

| 物体 | 动作 | 溢出(FP/pos) | 精度 |
|---|---|---:|---:|
| Mug | wrap_grasp | **12.38** | 0.037 |
| Door | pull | 7.28 | 0.049 |
| Scissors | grasp | 6.28 | 0.069 |
| Mug | pourable | 6.10 | 0.111 |
| Knife | cut | 5.98 | 0.132 |

**Mug 的 `wrap_grasp` 溢出 12 倍**（真实 688 点，误报 8520 点）——把手周围整片被激活。**Door 的 `pull` 溢出 7.3 倍**（真实 327 点）——门板大面积被当成把手。

### 3.4 新旧类抢占

新类样本上，GT 动作点被预测成其它动作的 Top 10：

| GT | → 预测 | 点数 |
|---|---|---:|
| support | layable | 48,585 |
| support | wear | 36,268 |
| support | listen | 35,076 |
| support | contain | 31,643 |
| contain | wear | 31,478 |
| support | sittable | 29,811 |
| contain | sittable | 29,347 |
| move | cut | 26,900 |
| support | pull | 25,862 |
| contain | listen | 24,585 |

**`support` 和 `contain` 是最大的被抢占源**。这两个动作的 GT 区域都是"大面积平面/内腔"，会被 `layable/wear/listen/sittable` 这类大面积动作同时激活。

这不是单纯的"旧类抢新类"，而是**大面积动作之间的语义混淆**：模型对"平面就是 support 还是 layable/sittable"区分不开。

---

## 4. 最差样本与 ply

5 个最差新类样本**全部是 Vase**，选出的动作是 `contain`（第 2 个动作），IoU 全 0：

| idx | 点云 | GT 正例点 | 预测正例点 | IoU |
|---:|---:|---:|---:|---:|
| 121 | 2048 | 48 | 80 | 0.000 |
| 132 | 2048 | 33 | 79 | 0.000 |
| 145 | 2048 | **14** | 110 | 0.000 |
| 151 | 2048 | 37 | 13 | 0.000 |
| 162 | 2048 | 18 | **608** | 0.000 |

**GT 只有 14–48 个点**（占 2048 的 0.7%–2.3%）——`contain` 在花瓶上只标注瓶口内壁那一小圈。模型要么完全没找到（13 点），要么大面积溢出（608 点），**没有一次落在正确位置**。

文件在 `/data01/jh/afford_diag/ply/`：
`worst_121_Vase_a01_iou0.000.ply` 等 5 个，配色为 GT 绿 / 预测红 / 重叠黄 / 其余灰。

---

## 5. full vs partial 对照（同为 k5_seed1）

| 指标 | full | partial | 变化 |
|---|---:|---:|---:|
| 样本数 | 2285 | **9140** | 每个物体多视角 |
| novel 召回 | 0.426 | **0.337** | −8.9pp |
| base 召回 | 0.768 | **0.683** | −8.5pp |
| novel 溢出 | 1.918 | 1.902 | −0.016 |
| base 溢出 | 1.754 | 1.858 | +0.104 |
| 无效对误报占比 | 85.6% | 84.5% | −1.1pp |

**残缺视角几乎只影响召回，不影响溢出。** 这符合物理直觉：看不到的部分无法激活，但看得到的部分该误报还是误报。

### 5.1 逐物体类掉分（IoU = TP/(TP+FP+FN)，full → partial）

| 类别 | IoU full | IoU partial | Δ |
|---|---:|---:|---:|
| Display | 0.441 | 0.324 | **−0.117** |
| Laptop | 0.508 | 0.404 | **−0.104** |
| Dishwasher | 0.257 | 0.160 | −0.097 |
| Keyboard | 0.090 | 0.029 | −0.061 |
| TrashCan | 0.208 | 0.153 | −0.055 |
| Clock | 0.140 | 0.093 | −0.046 |
| Bag | 0.233 | 0.190 | −0.043 |
| Microwave | 0.139 | 0.096 | −0.042 |
| Chair | 0.290 | 0.250 | −0.040 |
| Vase | 0.146 | 0.110 | −0.036 |
| Bottle | 0.155 | 0.121 | −0.034 |
| Bed | 0.251 | 0.274 | **+0.023** |
| Scissors | 0.119 | 0.136 | **+0.017** |

**Display / Laptop / Dishwasher 掉最多**——都是"有一面关键功能区"的物体：屏幕、键盘面、洗碗机门。单视角很容易正好看不到那一面。

Bed 和 Scissors 反而涨了——这两个类在部分视角下可能恰好过滤掉了大量背景干扰。

### 5.2 逐动作掉分

| 动作 | IoU full | IoU partial | Δ |
|---|---:|---:|---:|
| **displaY** | 0.441 | 0.296 | **−0.146** |
| lift | 0.318 | 0.224 | −0.094 |
| press | 0.257 | 0.181 | −0.075 |
| sittable | 0.444 | 0.369 | −0.075 |
| contain | 0.231 | 0.184 | −0.047 |
| support | 0.196 | 0.152 | −0.044 |
| openable | 0.103 | 0.068 | −0.035 |
| grasp | 0.157 | 0.137 | −0.020 |
| move | 0.130 | 0.132 | +0.002 |

**`displaY` 掉 14.6pp**，`press` 掉 7.5pp，`openable` 掉 3.5pp——全是"朝向单一、只能从特定角度看"的功能面。而 `move`/`pourable`/`stab` 几乎不掉（它们依赖整体形状或尖端，不依赖特定视角）。

---

## 6. 判断

### 6.1 这个基线的问题在哪

三条，按可操作性排序：

1. **没有物体—动作有效性建模（85.6% 的误报）**。这是最大、最干净的一块。模型对 `(Chair, cut)`、`(Table, weep)` 这类无效组合毫无抵抗力。而 validity 在 manifest 里是**现成的标注**（`validity: {object: [affordances]}`），只是训练时没当监督用。
2. **大面积动作互相抢占**（`support`/`contain` 被 `layable/wear/sittable` 抢）。这是"切片式"监督的固有缺陷：同一批点可以支持多个动作，模型学会了对平面一律高激活。
3. **新类召回 0.426 是天花板**。但注意：从 1-shot 到 5-shot 只涨 1.94pp，说明这个天花板不是"支持样本不够"，而是**特征本身对功能区的定位能力不足**。

### 6.2 对换方向的启示

如果想在这个方向做方法，**最值得做的不是"更好的增量策略"，而是"物体—动作有效性"这一层**：

- 它是 85.6% 误报的来源
- 它有现成标注（不需要新数据）
- 现有 SOTA（GREAT/GEAL/MIFAG）都没碰这一点——它们解决的是增量遗忘和语言引导

另一个可做的点是 **partial 的视角补全**：`displaY` 掉 14.6pp，本质是"看不到就猜不到"。但这个方向的成本高（需要多视角融合），收益受限于数据集本身。

**不建议**的方向：继续在"1-shot vs 5-shot"上做文章（只差 1.94pp），或者"支持集掩码精度"（弱标注实验已证明只差 1–2pp）。

---

## 附：产物清单（全部在 `/data01/jh/afford_diag/`）

| 路径 | 内容 |
|---|---|
| `eval_k5_seed1/` | 官方 evaluate 输出（metrics.json / complete.json） |
| `full_k5/per_pair_errors.json` | 每个 (class, aff) 的 TP/FP/FN/召回/精度 |
| `full_k5/invalid_pair_errors.json` | 无效物体-动作组合的误报 |
| `full_k5/action_confusion.json` | 新类动作混淆矩阵 |
| `full_k5/worst_samples.npz` | 最差 5 样本的 coord/gt/pred |
| `ply/` | 5 个 Vase 的 ply（GT 绿 / 预测红 / 重叠黄） |
| `partial_k5/` | partial 协议的同样四套输出 |
| `compare_full_partial.json` | 逐类/逐动作的 full vs partial 对比 |
| `split_definition.json` | base/novel 划分 |

SSH 侧脚本：`/data01/jh/afford_error_diag.py`、`/data01/jh/afford_summary.py`、`/data01/jh/compare_full_partial.py`、`/data01/jh/dump_worst_ply.py`
