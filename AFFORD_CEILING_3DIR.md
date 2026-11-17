# 可供性基线：Oracle 上限、全监督上界与 3DIR 误差解剖

日期：2026-11-16
约束：`/nas_data/ljy/` 全程只读；所有输出在 `/data01/jh/afford_diag/`。

---

## 0. 结论速览

1. **partial 的 Oracle A 严格为 0，可实现增益只有 1–2pp。** 与 full 完全一致：无效配对置零对官方指标没有任何影响（58 个有效对之外根本不进指标）；单一全局阈值 +0.26pp，每类阈值 +1.22pp（新类 +2.43pp）。剩下的 13–16pp 缺口不是阈值/校准能填的。
2. **3DIR 上 APGraph 的 30.72 分，本质是「Base 不遗忘」，不是「新动作学得好」。** 按动作拆开：Base 9 个动作 aIOU **43.32**、Novel 8 个动作 **16.54**。四个图像/跨模态方法的 Novel 是 15.5–18.1（GEAL 18.10 还比 APGraph 高），但它们的 Base 只剩 12–13。
3. **3DIR 的低分不是「跨模态对齐失效」。** IAGNet/GEAL/GREAT/MIFAG 在增量开始前（stage0）Base 动作 aIOU 是 39–41、AUC 85–88 —— 图文-点云对齐本身是可用的。**第一个增量阶段就把 Base 打到 3.6–8.6**，之后再也回不来。这是灾难性遗忘。
4. **APGraph 也不是「定位不到」：是「区域面积/边界不准」。** 平均 AUC 84.4，但 aIOU 只有 30.7。`Knife-stab` 的 AUC 是 96.8，aIOU 11.3；`Knife-cut` 81.2 / 12.1；`Suitcase-pull` 95.8 / 27.3。排序几乎完美，预测区域与 GT 的面积/位置对不上。
5. **少数组合是真分不开**：`Knife-grasp` AUC 47.4 / aIOU 0.71、`Scissors-grasp` 43.1 / 0.89、`Baseballbat-grasp` 31.3 / 4.60。细长工具的手柄区域，模型完全没有概念。
6. **`Knife-stab` 的根因是概率饱和，不是 GT 太小。** 同一个 query 面对同样的 7 个 GT 点，APGraph 的 p90 已经是 1.000，阈值提到 0.99 还剩 255 个预测点（248 个假阳性）；IAGNet 的 p90 是 0.066，t=0.90 只激活 16 个点。AUC 对「一小撮饱和并列」不敏感，aIOU 对「任何面积溢出」极其敏感。
7. **APGraph 的防遗忘门不是新动作差的原因。** 把每个 expansion 的历史标准化边界（`bias=0 / tau_vis=1`）关掉，Novel 8 动作只涨 **0.38pp**、Base 9 动作掉 **11.0pp**；`grasp`/`cut`/`wrapgrasp` 关掉后反而更差。`Knife-grasp` 和 `Bottle-wrapgrasp` 的 GT 激活概率本来就是 0，不是被压到 0 的。
8. **图像信息的价值只在「有明确视觉形状的握持区」**：`Mug-grasp`（把手环）IAGNet 41.19 / AUC 96.1 vs APGraph 4.30 / 65.0；`Baseballbat-grasp` 10.37 vs 4.60。但 `Knife-grasp`/`Scissors-grasp`/`Bottle-wrapgrasp` 两边都废，APGraph 甚至更好。
9. **全监督 23 类的上界是 44.60 mAP / 24.94 maIOU（full），APGraph 差 6.00 / 4.92 —— 全部来自 Novel 30 对。** Base 28 对上 APGraph（43.87）反而比全监督（41.64）高 2.23pp，因为全监督被数据不平衡稀释。**「增量顺序」不是瓶颈，「5-shot 支持」才是（Novel 差 13.61pp）。**
10. **partial 上 few-shot 损失只剩 1.62pp，但全监督本身掉了 10.18pp**（44.60 → 34.42）。**单视角缺失比 5-shot 更根本**：连全监督都只有 34.42，APGraph 的 32.80 已经接近这个上限。
11. **Base 属性在给 Novel 动作「注水」，而不是帮忙。** `z = (response*alpha) @ a` 对全部 145 个属性求和，Base 127 个属性的响应直接进入 Novel 的 logit。把 Base 的视觉门全关，Novel 8 从 15.87 掉到 **10.64（−5.23pp）**；而 `move`（8.91→11.92）、`grasp`（5.46→5.62）、`stab`（8.49→9.27）关掉 Base 后**反而更好**。Base 的大面积偏好（support 40%、play 20.7%）通过这个求和把 Novel 推向大面积预测，这解释了面积相关性 rho=0.497。
12. **温度缩放/偏置校准的增益是 0。** 在 Base 9 上选到的最优 `(c, T)` 就是 `(0, 1)`，任何 `T>1` 或 `c≠0` 都让 Novel 8 变差。原因是 **aIOU 是 20 档阈值的平均，对任何单调变换近似不变**。「±31 饱和」在官方指标下不是可回收的损失，只影响固定 0.5 阈值的部署。
13. **公开 CL 基线里，EWC / FrozenHead 是「Base 半遗忘 + Novel 正常」，不是「保 Base 压垮 Novel」。** 5-shot 最终：EWC Base 26.18 / Novel 16.70、FrozenHead 25.62 / 14.27、LwF 10.84 / 17.79、FT 11.74 / 17.38；APGraph 43.32 / 16.54。LwF 的蒸馏完全没阻止遗忘。分阶段上 EWC 是唯一 Old 稳定在 23–30 的方法。

---

## 1. partial 的 Oracle 对照

`partial_k5`（单视角残缺，9140 样本，k5_seed1，官方 58 个有效对）：

| variant | 自由度 | All mAP | All maIOU | New maIOU | novel12 maIOU |
|---|---:|---:|---:|---:|---:|
| raw | 0 | 32.80 | **16.55** | 14.77 | 13.17 |
| Oracle A（无效对置零） | — | 32.80 | **16.55** | 14.77 | 13.17 |
| global（1 阈值） | 1 | 32.80 | **16.81** | 15.87 | 14.02 |
| perclass（23 阈值） | 23 | 32.80 | **17.77** | 17.20 | 15.05 |

对比 full（`oracle_full_k5.json`）：

| variant | full All | full New | full novel12 |
|---|---:|---:|---:|
| raw | 20.02 | 20.25 | 16.60 |
| global | 20.42 (+0.40) | 19.30 (**−0.95**) | 16.44 (−0.16) |
| perclass | 22.02 (+2.00) | 21.49 (+1.24) | 18.17 (+1.57) |

**partial 的 perclass 增益（+1.22pp all / +2.43pp new）比 full 略大**——因为 partial 下整体概率偏低（Keyboard 最优阈值 0.02、Table 0.12、Clock/Display 0.14）。但仍然是 1–2pp 量级，而且 full 的 global 在新类上掉分、partial 的 global 在新类上 +1.10pp，**方向相反**，说明这层校准没有可迁移规律。

---

## 2. 3DIR / PIAD 误差解剖

### 2.1 口径核对

官方 `code/fbox/metrics.py` 的 aIOU 定义：

```python
ious = [np.logical_and(p >= t, z).sum() / np.logical_or(p >= t, z).sum()
        for t in np.linspace(0, 1, 20)]
aIOU = 100 * np.mean(ious)
```

**aIOU 是 20 档阈值（0→1）IoU 的平均，不是 0.5 单点 IoU。** 我用 `stage4_query.npz` 的 `pred/target` 按同口径逐对重算，三个有 npz 的方法与官方逐动作表**完全对齐**：APGraph 30.72 = 30.72、IAGNet 14.40 = 14.40、GREAT 14.13 = 14.13。

GEAL / MIFAG 没有逐对 dump，只进了逐动作表（来自 `stage4_actions.json`）。

**aIOU 的数值尺度受 GT 面积强烈影响**：3DIR 各动作的 GT 正例点占比中位数（2048 点中）：

| 动作 | GT 占比中位数 | 动作 | GT 占比中位数 |
|---|---:|---|---:|
| stab | **0.5%** | sit | 9.1% |
| pull | 2.1% | move | 9.2% |
| lift | 2.3% | lay | 11.0% |
| wrapgrasp | 2.6% | carry | 14.4% |
| press | 3.4% | play | 20.7% |
| open | 3.9% | support | **40.0%**（最大 91.7%） |
| grasp | 4.0% | | |
| cut | 4.1% | | |

**8 个动作的 GT 面积 < 5%，3 个动作 > 14%。** 小面积目标对任何溢出都极度敏感：`stab` 的 GT 只有 7–14 个点，多标 240 个点就把 IoU 打到 2%。所以 3DIR 的 aIOU 宏平均天然被小面积动作拖低——这不是方法独有的缺陷，但也不能拿它当借口：`support` 40% 面积只做到 60.8，`play` 20.7% 做到 76.4，说明大面积动作还有 24–39pp 的空间。

**这个面积依赖在三个方法上的表现不同**（Spearman，31 个 (object, action) 对）：

| 方法 | aIOU vs GT 面积 | AUC vs GT 面积 |
|---|---|---|
| APGraph | **rho = 0.497** (p=0.0045) | 0.095 (p=0.61) |
| IAGNet | 0.051 (p=0.79) | −0.153 (p=0.41) |
| GREAT | −0.003 (p=0.99) | −0.117 (p=0.53) |

**APGraph 的 aIOU 与 GT 面积中等正相关，图像方法没有这个相关。** 也就是说 APGraph 的 30.72 有相当一部分来自"大面积动作做得好"（`support`/`play`/`carry`），在小面积动作上和图像方法一样无能为力；图像方法则是"各面积都不好，但没有面积偏差"。

作为对照：**3D_AffordanceNet 上没有这个面积依赖**（58 个有效对的 recall vs 正例点数 rho = −0.074，p=0.58）。所以面积依赖是 3DIR 的极端 GT 分布（0.3%–91.7%）带来的，不是 aIOU 指标的普遍性质。

### 2.2 5-shot × 5 seed：Base 9 动作 vs Novel 8 动作

| 方法 | Base 9 动作 aIOU | Novel 8 动作 aIOU | All 17 动作 |
|---|---:|---:|---:|
| **APGraph** | **43.32** | 16.54 | **30.72** |
| IAGNet | 12.37 | 16.67 | 14.40 |
| GEAL | 13.07 | **18.10** | 15.44 |
| GREAT | 12.13 | 16.37 | 14.13 |
| MIFAG | 12.76 | 15.49 | 14.04 |

**Novel 上 APGraph 只排第三（GEAL 18.10 > IAGNet 16.67 > APGraph 16.54）。它的 30.72 全部来自 Base 9 个动作的 43.32。**

Novel 8 动作内部的逐物体分布（APGraph，aIOU 从低到高）：

| 物体 | novel 动作 aIOU | 用到的 novel 动作数 |
|---|---:|---:|
| Mug | 3.32 | 2 |
| Bowl | 3.46 | 1 |
| Vase | 3.61 | 1 |
| Baseballbat | 4.60 | 1 |
| Umbrella | 5.47 | 1 |
| Knife | 8.03 | 3 |
| Tennisracket | 9.29 | 1 |
| Scissors | 9.41 | 2 |
| Chair | 11.62 | 1 |
| Bag | 16.50 | 1 |

**全是"小件 / 细长 / 容器"**。有意思的是 `Chair`：它的 novel 动作只有 11.62，但 base 动作有 44.63 —— **物体本身见得多，不代表新动作能落地**。

### 2.3 逐动作 aIOU（%）

| 动作 | 分组 | APGraph | IAGNet | GEAL | GREAT | MIFAG |
|---|---|---|---:|---:|---:|---:|---:|
| play | base | **76.44** | 8.15 | 4.96 | 5.20 | 8.42 |
| support | base | **60.83** | 5.90 | 8.56 | 5.60 | 4.91 |
| ride | base | **50.21** | 6.70 | 7.49 | 7.04 | 10.16 |
| lift | base | **47.97** | 27.58 | 32.95 | 28.90 | 33.24 |
| carry | base | **46.96** | 9.97 | 11.20 | 10.44 | 10.47 |
| open | novel | 38.47 | **46.78** | 42.85 | 41.63 | 43.00 |
| pour | base | **31.24** | 30.72 | 27.14 | 28.47 | 27.64 |
| sit | base | **31.09** | 2.99 | 3.47 | 2.67 | 2.66 |
| listen | base | **30.65** | 18.21 | 19.78 | 19.39 | 16.66 |
| pull | novel | 27.26 | 35.60 | 35.87 | **39.00** | 37.20 |
| lay | novel | **18.93** | 2.78 | 2.91 | 2.60 | 3.21 |
| cut | novel | 14.82 | 12.06 | **19.95** | 14.73 | 8.53 |
| press | base | **14.48** | 1.15 | 2.05 | 1.45 | 0.66 |
| move | novel | **11.62** | 2.26 | 3.57 | 2.23 | 3.05 |
| stab | novel | 11.26 | 17.23 | **18.87** | 16.39 | 13.12 |
| grasp | novel | 5.85 | 14.88 | **19.05** | 12.38 | 14.05 |
| wrapgrasp | novel | **4.09** | 1.80 | 1.78 | 2.04 | 1.75 |

**两边的强项几乎互补**：APGraph 在 `play/support/ride/carry/sit/move` 这类「整体形状可判定」的动作上碾压（+37～+71pp）；图像方法在 `open/pull/grasp/stab/cut` 这类「局部功能件」上更好（+3～+13pp）。

### 2.4 逐物体类 aIOU（%，有逐对数据的三个方法）

| 物体 | APGraph | IAGNet | GREAT |
|---|---:|---:|---:|
| Skateboard | **83.45** | 9.86 | 9.11 |
| Guitar | **76.44** | 8.15 | 5.20 |
| Motorcycle | **52.27** | 3.66 | 5.91 |
| Bicycle | **47.84** | 10.16 | 8.32 |
| Backpack | **46.50** | 20.10 | 20.76 |
| Bag | **44.27** | 25.51 | 26.78 |
| Surfboard | **41.59** | 1.23 | 1.49 |
| Bottle | 31.52 | **34.35** | 31.09 |
| Suitcase | 27.26 | 35.60 | **39.00** |
| Earphone | **30.65** | 18.21 | 19.39 |
| Chair | **23.10** | 1.87 | 1.76 |
| Bed | **17.87** | 3.52 | 3.27 |
| Scissors | 14.14 | 12.71 | 11.48 |
| Keyboard | **14.48** | 1.15 | 1.45 |
| Knife | 10.69 | 13.02 | **14.56** |
| Tennisracket | 9.29 | 16.92 | **23.13** |
| Umbrella | 5.47 | **12.31** | 8.60 |
| Baseballbat | 4.60 | **10.37** | 1.28 |
| Vase | **3.61** | 2.90 | 3.08 |
| Bowl | 3.46 | 2.42 | 3.08 |
| Mug | **3.32** | 20.66 | 16.30 |

**APGraph 的强项是「大件、整体形状独特」的物体**（Skateboard/Guitar/Motorcycle/Bicycle），弱项是「小件、局部功能区域」的物体（Mug/Bowl/Vase/Baseballbat/Umbrella/Tennisracket）。

### 2.5 错题性质：排序能力 vs 区域精度

APGraph 平均 AUC **84.4**，aIOU **30.7**。逐组合的差距：

| 组合 | AUC | aIOU | bestIOU（20 档里最好） | 差 |
|---|---:|---:|---:|---:|
| Knife-stab | 96.78 | 11.26 | 21.23 | 85.5 |
| Knife-cut | 81.21 | 12.10 | 19.38 | 69.1 |
| Suitcase-pull | 95.78 | 27.26 | 37.13 | 68.5 |
| Bed-sit | 83.37 | 15.61 | 20.19 | 67.8 |
| Keyboard-press | 81.97 | 14.48 | 18.52 | 67.5 |
| Bowl-wrapgrasp | 68.31 | 3.46 | 5.53 | 64.9 |
| Scissors-cut | 82.42 | 17.93 | 23.30 | 64.5 |
| Mug-grasp | 65.01 | 4.30 | 8.35 | 60.7 |

**即使把阈值扫到最优（bestIOU），这些组合也只有 5–37。** 不是阈值问题，是**预测区域的形状/面积与 GT 不匹配**：模型知道"大概在刀刃上"，但标出来的区域比真实区域大一个量级，或者整体偏移。

真正"连排序都不行"的组合（AUC 也低）：

| 组合 | AUC | aIOU |
|---|---:|---:|
| Baseballbat-grasp | 31.32 | 4.60 |
| Scissors-grasp | 43.12 | 0.89 |
| Knife-grasp | 47.44 | 0.71 |
| Bottle-wrapgrasp | 49.37 | 6.51 |

这四组都是**细长工具/容器的"握持区"**——GT 是手柄/瓶身的一小段，模型把整根杆/整个瓶身都激活了。

### 2.6 图像方法的崩点在增量，不在跨模态

各阶段 Base 9 动作的平均 aIOU 与 AUC（5 seed 平均）：

| 方法 | stage0 | stage1 | stage2 | stage3 | stage4 |
|---|---:|---:|---:|---:|---:|
| IAGNet aIOU | **40.76** | 7.12 | 8.83 | 5.62 | 12.37 |
| IAGNet AUC | **87.62** | 60.31 | 65.88 | 60.62 | 56.95 |
| GEAL aIOU | **40.25** | 8.60 | 10.50 | 4.12 | 13.07 |
| GEAL AUC | **87.56** | 70.97 | 65.73 | 61.06 | 59.25 |
| GREAT aIOU | **40.00** | 3.59 | 6.75 | 3.34 | 12.13 |
| GREAT AUC | **84.89** | 53.22 | 61.16 | 54.00 | 62.52 |
| MIFAG aIOU | **39.05** | 4.91 | 6.62 | 4.70 | 12.76 |
| MIFAG AUC | **86.95** | 56.94 | 56.87 | 52.96 | 59.00 |

**stage0 的 AUC 85–88 直接排除了"图文-点云对齐失效"这个解释**：如果对齐坏了，base 动作的排序不可能这么高。**第一个增量阶段（grasp/wrapgrasp）一训，Base 的 aIOU 掉 32–37pp、AUC 掉 17–32pp**，之后四个阶段在低水平震荡。

APGraph 的对照：stage0 = stage4 = 43.37（逐动作完全一致），因为它的增量只给新动作加接收者节点，不碰 Base 的输出路径。

### 2.7 功能框大小/遮挡的混淆检验

3DIR 的每张查询图都带一个 `functional_bbox_xyxy`（人手所在的功能区域）。如果"人手遮挡或图文姿态不一致"是主因，功能框越小、指标应该越差。

实测（255 对，Spearman）：**方向相反**。

| 方法 | AUC vs 功能框面积占比 | aIOU vs 功能框面积占比 |
|---|---|---|
| IAGNet | **rho = −0.393** (p=8e-11) | −0.342 (p=2e-8) |
| GREAT | **rho = −0.415** (p=5e-12) | −0.365 (p=2e-9) |

按功能框面积四分位分组（小→大）的 IAGNet AUC：**78.50 / 60.86 / 54.81 / 53.56**；GREAT：83.52 / 64.23 / 60.11 / 57.56。

**功能框越小（局部精细功能件），图像方法反而做得越好；框越大（lay/ride/play 这类大面积动作）越差。** 这与"遮挡导致失效"的假设完全相反。

注意这个相关主要反映**动作难度**而不是图像质量：`pull` 的功能框中位数 0.0027（最小），IAGNet aIOU 35.60；`lay` 是 0.0788（最大），IAGNet aIOU 2.78。但无论怎么解释，**数据不支持"图像质量/遮挡是 3DIR 低分主因"**。

### 2.8 Knife-stab 的饱和解剖：为什么 AUC 96.8 而 aIOU 11.3

同一批 query（同样的 7–14 个 GT 点，占 2048 的 0.34%–0.68%），APGraph 与 IAGNet 的逐点分布：

| 指标 | APGraph | IAGNet |
|---|---|---|
| GT 正例 | 7 / 2048（0.34%） | 7 / 2048（0.34%） |
| pred p50 | 0.000 | 0.002 |
| pred **p90** | **1.000** | 0.066 |
| pred p95 | 1.000 | 0.154 |
| pred p99 | 1.000 | 0.346 |
| GT+ 平均概率 | 1.000 | 0.999 |
| GT− 平均概率 | 0.159 | 0.026 |
| t=0.90 预测点数 | **283**（TP 7 / FP 276） | **16**（TP 7 / FP 9） |
| t=0.99 预测点数 | **255**（TP 7 / FP 248） | 11（TP 7 / FP 4） |
| aIOU（20 档平均） | 0.023 | **0.278** |
| 最优单阈值 IoU | 0.071 | 0.438 |

**答案是 (a) 预测饱和 + (b) GT 极小，两者叠加，而主因是 (a)。**

- APGraph 把**整片刀身预测成 1.000**：p90 就已经是 1.0，阈值提到 0.99 仍有 255 个点（其中 248 个是假阳性）。GT 点概率确实是 1.0（召回 100%），但一大片负例也是 1.0 —— **排序无法区分，阈值也无法切细**。
- IAGNet 的概率有梯度（p90=0.066、p99=0.346），t=0.90 时只激活 16 个点，命中 7 个 —— 这才是"定位到刀尖"。
- **AUC 对这种饱和并列不敏感**：GT 点都在最高分段，AUC 就高；但那 248 个并列 1.0 的负例点不改变 rank 统计，却把 IoU 压到 2%。

逐点 IoU 曲线（20 档阈值）也印证：APGraph 的曲线从 0.00 平到 0.07，**没有可用的工作点**；IAGNet 的曲线在 t≈0.7 处升到 0.44。

同样的模式出现在 `Knife-cut`（AUC 81.2 / aIOU 12.1）、`Suitcase-pull`（95.8 / 27.3）、`Bed-sit`（83.4 / 15.6）、`Keyboard-press`（82.0 / 14.5）。

**这个饱和是 3DIR 特有的，不是 APGraph 的通病。** 在 3D_AffordanceNet 的 58 个有效对上，p90 > 0.9 的对数是 **0 / 58**，最高的是 `Laptop-displaY` 0.873、`Chair-sittable` 0.738；`>=0.99` 的点占比最高只有 0.4%（`Bag-lift`）。也就是说 3D_AffordanceNet 上的阈值 0.5 是合理的，3DIR 上的输出尺度退化了。

### 2.9 「握持区」四对：图像方法是不是真会、APGraph 是不是被抑制压坏

#### (1) 四对 + 两个对照的完整指标（5 seed 逐对平均）

| 配对 | APGraph aIOU / AUC | IAGNet aIOU / AUC | GREAT aIOU / AUC |
|---|---|---|---|
| Baseballbat-grasp | 4.60 / 31.32 | **10.37 / 55.59** | 1.28 / 59.23 |
| Knife-grasp | **0.71 / 47.44** | 0.23 / 32.37 | 0.16 / 36.08 |
| Scissors-grasp | **0.89 / 43.12** | 0.05 / 45.78 | 0.06 / 34.30 |
| Bottle-wrapgrasp | **6.51 / 49.37** | 1.33 / 49.49 | 1.39 / 47.50 |
| Mug-grasp（对照） | 4.30 / 65.01 | **41.19 / 96.11** | 32.47 / 94.60 |
| Umbrella-grasp（对照） | 5.47 / 56.95 | **12.31 / 73.51** | 8.60 / 67.49 |

**不是"图像方法都会做、APGraph 被忘了"**，而是分物体：
- `Mug-grasp`（把手是封闭环）：IAGNet 41.19 / AUC 96.1，APGraph 4.30 / 65.0 —— **差 37pp**
- `Baseballbat-grasp`（握柄有粗细变化）：IAGNet 10.37 / 55.6，APGraph 4.60 / 31.3
- `Umbrella-grasp`：IAGNet 12.31 / 73.5，APGraph 5.47 / 57.0
- 但 `Knife-grasp`、`Scissors-grasp`、`Bottle-wrapgrasp`：**两边都废，APGraph 甚至更好**（0.71/0.89/6.51 vs 0.23/0.05/1.33）

图像信息的价值集中在「**有明确视觉形状的握持区**」（环、粗细节），对「被手挡住或本身没有视觉边界的握持段」（刀柄、剪刀柄、瓶身）没有帮助。

#### (2) 关掉 APGraph 的历史边界校准门：novel 只涨 0.4pp，base 崩 11pp

APGraph 的每个新动作 expansion 有一个「历史标准化视觉门」：

```python
# continual/expansion.py initialize()
offset = -float(cfg['boundary_value'])          # -20
tau_vis = scale                                  # 历史点云在 direction 上的投影标准差
exp.bias = offset - (stats['mean'] @ unit.T) / tau_vis
# visual_activation(e) = 0.5 * (1 + erf(((e - mean) @ unit / tau_vis - 20) / sqrt(2)))
```

即：一个点要偏离**历史 base 点云均值 20 个标准差**才会激活新动作。这是极保守的防遗忘门。

同一 stage4 权重，三组对照（seed1，255 对，逐对重算）：

| variant | Base 9 动作 aIOU | Novel 8 动作 aIOU | mean AUC |
|---|---:|---:|---:|
| raw（原始门控） | **43.14** | 15.87 | 84.12 |
| gate_off（bias=0，tau=1） | **32.13（−11.0）** | **16.25（+0.38）** | 80.67 |
| zero_new（新动作全置零） | 43.09 | 13.90 | 80.44 |

**关掉门控后，Novel 只涨 0.38pp，Base 掉 11.0pp。** 逐动作更直观：

| 动作 | raw | gate_off | 变化 |
|---|---:|---:|---:|
| open | 42.33 | 50.81 | **+8.5** |
| move | 8.91 | 10.82 | +1.9 |
| stab | 8.48 | 9.61 | +1.1 |
| **grasp** | 5.47 | 4.73 | **−0.7** |
| **cut** | 12.41 | 9.69 | **−2.7** |
| **wrapgrasp** | 4.95 | 2.32 | **−2.6** |
| carry | 46.96 | 23.04 | −23.9 |
| ride | 50.23 | 27.87 | −22.4 |
| sit | 31.08 | 11.62 | −19.5 |
| support | 60.83 | 38.10 | −22.7 |

**新精细动作根本没有被"压坏"**：关掉门控后 `grasp`/`cut`/`wrapgrasp` 反而更差，只有 `open` 显著受益（+8.5pp）。而 base 的 `carry/ride/sit/support` 直接掉 20pp 以上。

关键配对的逐对结果也一致：

| 配对 | raw aIOU / AUC | gate_off aIOU / AUC |
|---|---|---|
| Knife-stab | 0.023 / 0.98 | 0.038 / 0.96 |
| Knife-grasp | 0.002 / 0.17 | 0.002 / 0.44 |
| Baseballbat-grasp | 0.016 / 0.61 | 0.015 / 0.43 |
| Bottle-wrapgrasp | 0.000 / 0.06 | 0.000 / 0.39 |

`Knife-grasp` 和 `Bottle-wrapgrasp` 的 aIOU 在开/关两种情况下都是 0.002 / 0.000 —— **GT 点的激活概率本来就是 0，不是被压到 0 的**。

**结论：新精细动作差是因为新几何属性本身没学好，不是防遗忘抑制的代价。**

另外 `zero_new` 显示：新动作对整体只贡献 1.97pp（15.87→13.90），对 Base 的抢占只有 0.05pp（43.14→43.09）。APGraph 的防遗忘是真的零成本——但代价是新动作也没学进去。

### 2.10 握持配对的 stage1→stage4 轨迹：APGraph「从没学会」，IAGNet「学了又忘」

**先注意一个必须澄清的事实：两个方法的 stage 顺序不同。**

| 阶段 | IAGNet（`3dir_swap_grasp_wrapgrasp_with_lift_pour_v1`） | APGraph（`3dir_swap_stage_order_2143_v1`） |
|---|---|---|
| s1 | **grasp, wrapgrasp** | lay, move |
| s2 | lay, move | **grasp, wrapgrasp** |
| s3 | **cut, stab** | open, pull |
| s4 | open, pull | **cut, stab** |

主表的最终数字（17 个动作都学完）可比，但 **stage-by-stage 的曲线不可直接对比**。下面的 APGraph 列中，`grasp` 在 s2 才变好是符合它的协议顺序的。

seed1 的逐对重算（aIOU / AUC，%）：

| 配对 | APGraph s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| Baseballbat-grasp | 0.73 | 0.73 | **10.67** | 10.67 | 10.67 |
| Knife-grasp | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| Scissors-grasp | 0.48 | 1.61 | 0.07 | 0.07 | 0.07 |
| Bottle-wrapgrasp | 1.57 | 1.57 | **51.60** | 3.03 | 3.03 |
| Mug-grasp | 9.46 | 9.46 | 6.28 | 6.28 | 6.28 |
| Knife-stab | 5.17 | 5.10 | 5.10 | 5.10 | 8.63 |

| 配对 | IAGNet s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| Baseballbat-grasp | 0.73 | 0.73 | 0.73 | **40.98** |
| Knife-grasp | **10.84** | 0.10 | 0.10 | 0.10 |
| Scissors-grasp | 0.07 | 0.07 | 0.07 | 0.07 |
| Bottle-wrapgrasp | 2.95 | **22.05** | 1.57 | 1.57 |
| Mug-grasp | 25.07 | 0.20 | 5.06 | **61.83** |
| Knife-stab | 0.03 | 17.45 | **66.70** | 45.16 |

**IAGNet 在 stage1 确实学会了 `Knife-grasp`（aIOU 10.84 / AUC 99.52），stage2 就掉到 0.10/87.87，stage4 是 0.10/66.77。`Bottle-wrapgrasp` 在 stage2 还有 22.05，stage4 只剩 1.57。**

但这不是统一遗忘：同一时期的 `Mug-grasp`（25.07→61.83）、`Baseballbat-grasp`（0.73→40.98）、`Knife-stab`（0.03→45.16）反而越学越好。**IAGNet 的增量对不同配对的命运是分裂的**，取决于后续阶段的数据分布。

APGraph 没有"学了又忘"的机会：`Knife-grasp` 全程 0.10，`Scissors-grasp` 从 0.48 掉到 0.07。它的属性图扩展在 `Bottle-wrapgrasp` 上 stage2 冲到 51.60、stage3 又被打回 3.03 —— **新属性之间会互相干扰**，也不是稳定的单调改善。

GEAL 只有逐动作：`grasp` 7.66→13.36（涨）、`wrapgrasp` 3.96→1.64（降）。动作级平均掩盖了配对级的分化。

### 2.11 冻结范围与饱和归因

**冻结范围（`infer_graph.load_model`）**：

```python
model.requires_grad_(False)   # 整个模型：3D PointNet++ 骨干、2D DINOv2 ViT、
                              # base 属性图、grounding_head 全部冻结
```

**增量阶段只训 `expansion.parameters()`**（`continual/expansion.py: fit_expansion`）：

| 参数 | 含义 | stage1/2/3/4 节点数 |
|---|---|---|
| `a` | 新属性向量（192 维） | 7 / 4 / 4 / 3 |
| `no` | 新→旧边强度 | 同上 |
| `nn` | 新→新边强度 | 同上 |
| `radius_raw` | 半径 | 同上 |

`v` / `bias` / `tau_vis` / `on` / `mask` 都是 buffer，不参与训练。

**饱和归因：不是特征不可分，不是属性范数爆炸，也不是视觉门。**

对 Knife-stab query `7654eded`（GT 7 点）：

| 检查项 | 数值 | 判断 |
|---|---|---|
| GT+ ↔ 高分区 h 余弦 | **0.48** | 骨干特征**有**区分度，不是"完全一样" |
| expansion `a` 范数 | 3.5 / 4.1 / 4.0 / 3.5 | 比 base 的 **11.3 更小**，不是爆炸 |
| 视觉门 alpha（stage4 三节点） | **[1.0, 1.0, 1.0]** | 门全开；但 gate_off 无改善 |
| GT+ logit | mean **29.02**，std 0.82，范围 27.8–30.4 | |
| 高分区 253 个负例 logit | mean 16.02，范围 **4.67–31.16** | **最高 31.16 超过 GT 最高 30.38** |
| 其余点 logit | mean **−18.34**，std 9.76 | |

**根因是 logits 量级爆炸到 ±31**：

```python
response = h @ F.normalize(raw_attributes()).T   # h 范数 ~14
node = response * alpha                            # alpha = 1.0（全开）
z = node @ a                                       # a 范数 3.5
logits = grounding_head(z)                         # Linear(192→64) 权重范数 4.8
```

链式累乘，**中间没有任何归一化**。sigmoid 在 |logit| > 10 就完全饱和，整片刀身落进 25–31 的高分段，与 GT 的 27.8–30.4 混在一起。AUC 96.8 来自"大部分负例在 −18"，aIOU 2.3% 来自"253 个负例里有 246 个在 0.5 以上"。

**视觉门的机制**：`tau_vis` 的实际取值是 **0.0022–0.0043**（不是 1）。由于 `e` 和 `v` 都做过 L2 归一化，`e @ v ∈ [−1, 1]`，所以 `z_vis = (e@v)/0.0025 + bias ≈ ±400 − 12`，`erf` 完全饱和。**这个视觉门实际上是「查询级二值开关」**：图像 ROI 方向与某个历史节点一致就全图激活 1，否则全图 0。它不提供逐点区分度。`alpha` 全 1.0 就是这个开关打开的结果。

**注意：饱和不影响官方 aIOU。** 见 2.13 的温度校准实验——`aIOU` 是 20 档阈值的平均，对任何单调变换近似不变。饱和只影响"固定 0.5 阈值部署"时的面积，不影响论文指标。

**训练日志也证明是「支持集过拟合」而不是「没训够」**（`stage4_expansions.pt` 的 `audit.history`）：

| 指标 | step 1 | step 50 | step 550 | step 600 |
|---|---:|---:|---:|---:|
| ground loss | 11.30 | 5.98 | 2.76 | **1.81** |
| prototype norm | 0.99 | 1.63 | 2.99 | 3.00 |
| alpha_new_mean | 0.75 | 0.81 | 0.94 | 0.94 |

支持集 loss 降了 6 倍（11.3 → 1.8），但 query 上的 aIOU 只有 8.63。同时 `support_mean_abs_logit_drift = 10.51` —— 新属性把支持集上已有动作的 logit 平均改变了 10.5，说明它不是在学"刀尖的位置"，而是在学"支持集这几个样本的偏置"。

### 2.12 Base 属性通过共享读出给 Novel 动作「注水」

**前向拼接的完整路径**（`continual_grounder.py`）：

```python
# bank()
raw = cat([self.a] + [exp.prototypes() for exp in expansions])   # [127+7+4+4+3, 192]
g   = block_matrix([[frozen_g, on], [no, nn]])                    # on 全 0
enhanced = raw + g @ self.graph.value(raw)                        # [145, 192]
enhanced = cat([frozen_a, enhanced[127:]])                        # Base 部分强制冻结

# decode()
response = h @ normalize(raw_attributes()).T   # [N, 145]  用 raw 属性算匹配
node     = response * alpha[:, None, :]        # [N, 145]  视觉门是查询级 0/1
z        = node @ a                            # [N, 192]  Sum over ALL 145 attributes
logits   = self.grounding_head(z)              # Linear(192→64) + GELU + Linear(64→1)
```

**三个确认：**

1. **192 维怎么来的**：`z = node @ a` 里 `a` 是 `[145, 192]`，`node` 是 `[N, 145]`，求和后得到 `[N, 192]`。`h` 本身是 192 维（PointNet++ 的 `fp1` 输出），`raw_attributes()` 的每个属性也是 192 维（`dimension=192`）。**不是"h 拼接 2D 特征"，2D 特征只通过 `alpha` 的视觉门（查询级 0/1）进入。**
2. **Base 与 Novel 共用同一个 `grounding_head`**：`grounding_head = Sequential(Linear(192,64), GELU(), Linear(64,1))`，全模型冻结。Base 和 Novel 的动作只是在 `z` 的求和里用不同的 `a` 行，最后都过同一个 MLP。
3. **`no` 边和 `radius` 的作用**：
   - `radius`：`prototypes() = normalize(a) * sigmoid(radius_raw) 映射到 [0.3, 8]`，只影响 `a` 的尺度（`raw_attributes()` 里）。
   - `no`：`edge_max(0.3) * cross_strength(no_raw) * mask.T`，把**新属性从旧属性接收 value 加权和**（`enhanced[new] += no @ value(raw[old])`）。`on` 全 0，所以旧属性不接收新属性（`new_receivers_only_no_sender`）。

**验证：`z` 对全部 145 个属性求和，Base 的响应直接进入 Novel 的 logit。** 把 Base 127 个属性的视觉门全部置零：

| variant | Base 9 aIOU | Novel 8 aIOU |
|---|---:|---:|
| raw | 43.14 | **15.87** |
| `zero_old`（Base 门全关） | 2.91 | **10.64（−5.23）** |
| `zero_new`（新动作门全关） | 43.38 | 13.89 |

**Novel 8 有 5.23pp（约三分之一）来自 Base 属性的响应。** 逐动作：

| 动作 | raw | zero_old | 变化 |
|---|---:|---:|---:|
| open | 42.35 | 20.02 | −22.3 |
| pull | 26.31 | 14.69 | −11.6 |
| lay | 18.11 | 7.50 | −10.6 |
| **move** | 8.91 | **11.92** | **+3.0** |
| **grasp** | 5.46 | **5.62** | +0.2 |
| **stab** | 8.49 | **9.27** | +0.8 |

`open`/`pull`/`lay` 依赖 Base 的响应（Base 确实能帮忙），但 `move`/`grasp`/`stab` 关掉 Base 后**反而更好**——Base 的大面积响应在这些动作上是纯干扰。

**这解释了 2.1 里的面积相关性（rho = 0.497）**：Base 属性天然偏向大面积（support 40%、play 20.7%、carry 14.4%），它们的响应通过共享的 `z` 求和把 Novel 动作的预测推向大面积区域。**Novel 的低分不是"新属性学不会"，是"新属性的判断被 Base 的大面积偏好淹没了"。**

### 2.13 温度校准的零增益：aIOU 对单调变换不变

在 Base 9 上网格选全局 `(c, T)`，再原样应用到 Novel 8（不在 Novel 上挑参）：

```
在 Base9 上选到 c=0.0  T=1.0（Base9 43.14）
原样应用到 Novel8: 15.87
```

Novel8 的完整网格（行 = c，列 = T）：

| c \ T | 1 | 2 | 3 | 5 | 8 | 12 | 20 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| −20 | 8.96 | 8.49 | 8.34 | 8.62 | 9.04 | 8.65 | 7.36 | 6.23 |
| −10 | 12.60 | 12.01 | 11.94 | 11.76 | 10.80 | 9.37 | 7.32 | 6.00 |
| **0** | **15.87** | 15.10 | 14.36 | 12.79 | 10.81 | 8.99 | 6.92 | 5.65 |
| +10 | 7.86 | 8.12 | 8.41 | 8.54 | 8.14 | 7.33 | 6.07 | 5.12 |
| +20 | 2.54 | 2.65 | 2.83 | 3.45 | 4.48 | 5.03 | 4.89 | 4.52 |

**`c=0, T=1`（不校准）就是最高点，任何缩放/平移都变差。**

原因是数学上的：**aIOU 是 20 档阈值的平均，对任何单调变换近似不变**。`sigmoid((logit-c)/T)` 只改变 0.5 阈值落在原始 logit 轴的哪个位置，而 aIOU 已经把阈值从 0 扫到 1。所以：

- **"输出尺度校准负责的增益区间" = 0**
- 「±31 饱和」在官方指标下不是一个可回收的损失
- 要提升 Novel 8，只能改"预测区域本身"（特征/属性），不能改"输出尺度"

### 2.14 IAGNet 与 APGraph 的机制差异

| 维度 | IAGNet | APGraph |
|---|---|---|
| 3D 主干（PointNet++） | **可训** | 冻结 |
| 2D 骨干（DINOv2 ViT-B/14） | **full_finetune** | 冻结 |
| 跨模态融合（Cross_Attention / JRA / ARM） | 可训 | 冻结 |
| 训练参数 | 除 `decoder.cls_head` 外全部 | 4 组（`a` / `no` / `nn` / `radius_raw`） |
| 损失 | **Focal(γ=2, α=0.25) + Dice + 0.5·KL** | `WeightedFunctionalLoss`（BCE+Dice） |
| 增量预算 | 80 epoch，lr 1e-4，Adam | 600 步，lr 0.02，Adam |
| 增量数据 | `support_replay: "none_current_session_only"` | 只含当前 stage 的 2 个动作 |
| 特征维度 | `emb_dim=512`（3D/2D 融合后） | `dimension=192`（3D 单模态） |

**IAGNet 能学到 `Knife-grasp 10.84`、`Mug-grasp 61.83` 的原因**：
1. **主干在更新**（3D + 2D 全量微调），能做特征适配，不是靠冻结特征上的线性读出；
2. **Focal Loss（γ=2, α=0.25）+ Dice** 对难样本和小面积区域更友好，APGraph 的 BCE+Dice 会被大面积负例主导；
3. **有 KL 蒸馏**（两种几何配对的输出一致性），提供额外的正则。

**但 IAGNet 的 `none_current_session_only` 解释了它的遗忘**：stage2 再训 80 epoch 时，同一套可训参数被新动作的梯度覆写，且完全不看 stage1 的动作。stage1 的 `Knife-grasp 10.84` 在 stage2 掉到 0.10 就是这个机制。

**APGraph 的对照**：全冻结 + 只训新属性 → 旧动作零遗忘，但新动作也只能在冻结特征上做线性读出，学不到 `Mug-grasp` 的把手环。

### 2.15 公开 CL 基线在 3DIR 上的对照（EWC / LwF / FrozenHead / SequentialFT）

**注意：这四个方法不是套在 IAGNet 上，是套在 APGraph 的共享 base 上**（`base_shared_sha256 = a68d117298...`，与 APGraph 完全同一个权重，Base 初始 aIOU 43.37）。

| 方法 | 可训范围 | 正则 |
|---|---|---|
| **FrozenHead** | 只训 `shared grounding_head MLP 192-64-1`；encoder / A / V / graph 全冻结 | 无 |
| **EWC** | last3 DINO blocks + PointNet++ + V + bias + graph + MLP | Fisher 400，1024 样本/阶段 |
| **LwF** | 同 EWC | 蒸馏 T=2，weight=1 |
| **SequentialFT** | 同 EWC | 无 |

四个方法都是 `fixed 127 discovered Base attributes; no expansion` —— 没有属性扩展。

**5-shot 最终数字（5 seed 平均）**：

| 方法 | Base 初始 | Base 最终 | Novel 8 | All 17 | Base 遗忘 |
|---|---:|---:|---:|---:|---:|
| SequentialFT | 43.37 | 11.74 | 17.38 | 14.39 | 31.63 |
| LwF | 43.37 | 10.84 | 17.79 | 14.11 | 32.54 |
| FrozenHead | 43.37 | **25.62** | 14.27 | 20.28 | 17.76 |
| EWC | 43.37 | **26.18** | 16.70 | 21.72 | 17.20 |
| **APGraph** | 43.37 | **43.32** | 16.54 | **30.72** | **0.00** |

**结论：EWC / FrozenHead 是「Base 半遗忘 + Novel 正常」，不是「为保 Base 把 Novel 压垮」。**
- EWC 保住 60% 的 Base（26.18），Novel 还有 16.70，与 APGraph 的 16.54 持平
- FrozenHead 保住 25.62，Novel 14.27（低 2.3pp）—— 唯一有轻微 trade-off 迹象的
- **LwF 完全没阻止遗忘**（Base 10.84，与 FT 的 11.74 一样崩），Novel 反而最高（17.79）

**分阶段（seed0；New = 当前阶段新动作，Old = 已见动作）**：

| 方法 | s1 New/Old | s2 New/Old | s3 New/Old | s4 New/Old |
|---|---|---|---|---|
| EWC | 4.17 / **29.88** | 16.26 / **30.34** | 26.35 / 25.39 | 41.73 / 23.82 |
| FrozenHead | 4.56 / 17.10 | 11.62 / 17.39 | 12.46 / 7.74 | 41.38 / 18.13 |
| LwF | 4.46 / 12.33 | 15.61 / 11.98 | 21.26 / 5.74 | **57.45** / 10.47 |
| SequentialFT | 4.02 / 6.40 | 17.76 / 16.11 | 20.89 / 4.71 | **57.91** / 10.61 |

**EWC 是唯一一个 Old 稳定在 23–30 的方法**；FrozenHead 的 Old 从 17 掉到 7.74 再回 18（不稳定）；LwF/FT 的 Old 一路崩到 10。所有方法的 stage4 New 都很高（41–58，open/pull 容易），stage3 New 最低（12–26，cut/stab 难）。

**谱系**：冻结越彻底 → Base 保得越好、Novel 略低；微调越多 → Novel 越高、Base 越崩。**APGraph 是唯一跳出这个 trade-off 的（Base 43.32 + Novel 16.54）**，靠的是"完全冻结 + 新属性扩展"。但它的 Novel 也没有超过 EWC/LwF 的 16.7–17.8 —— 它赢在 Base。

### 2.16 同物跨阶段冲突与 Base 独有物体

按 **IAGNet 的协议顺序**（s1=grasp/wrapgrasp, s2=lay/move, s3=cut/stab, s4=open/pull），21 个物体的动作-阶段分布：

| 物体 | 动作（阶段） | 跨阶段冲突 |
|---|---|---|
| **Knife** | grasp(s1), stab(s3), cut(s3) | **跨三段** |
| **Bottle** | pour(s0), wrapgrasp(s1), open(s4) | **跨三段** |
| Scissors | grasp(s1), cut(s3) | 跨两段 |
| Bag | lift(s0), grasp(s1) | 跨两段 |
| Bed | sit(s0), lay(s2) | 跨两段 |
| Chair | sit(s0), move(s2) | 跨两段 |
| Mug | wrapgrasp(s1), grasp(s1) | 同阶段 |
| Baseballbat / Tennisracket / Umbrella | grasp(s1) | 无 |
| Bowl / Vase | wrapgrasp(s1) | 无 |
| Suitcase | pull(s4) | 无 |
| Backpack | carry(s0) | **Base 独有** |
| Bicycle | ride(s0) | **Base 独有** |
| Earphone | listen(s0) | **Base 独有** |
| Guitar | play(s0) | **Base 独有** |
| Keyboard | press(s0) | **Base 独有** |
| Motorcycle | ride(s0) | **Base 独有** |
| Skateboard | support(s0) | **Base 独有** |
| Surfboard | carry(s0), support(s0) | **Base 独有** |

**除了 Knife 和 Bottle，还有 Scissors / Bag / Bed / Chair 四对跨阶段冲突。**

**Base 独有物体 8 个（21 个中的 38%）**：Backpack, Bicycle, Earphone, Guitar, Keyboard, Motorcycle, Skateboard, Surfboard。它们在整个增量过程中**从未出现在任何阶段的训练数据里**。

**这对"仅靠当前阶段点云做跨查询蒸馏"是硬约束**：这 8 个物体的特征漂移没有任何增量数据能约束。IAGNet 的 Base 从 43 掉到 12，主要就是这 8 个物体在无人看管的情况下漂走了。**必须靠冻结 Base 输出路径（APGraph 的做法）或对 `out_head` / 粗尺度层加梯度正交投影**，不能只靠"当前阶段数据内的蒸馏"。

### 2.17 IAGNet 的框图抠取与张量维度

**`functional_box` 的抠取方式是 RoIAlign，不是 Mask Pooling。**

```python
def get_mask_feature(self, raw_img, img_feature, functional_box):
    box  = functional_box * scale                      # 归一化到特征图
    roi  = cat([indices, box], 1)
    obj_feature     = roi_align(img_feature, roi,  output_size=(4,4), sampling_ratio=4)  # 框内
    context_feature = roi_align(img_feature, full, output_size=(4,4))                    # 全图
    mask = ones_like(img_feature); mask[box region] = 0
    return obj_feature, context_feature, img_feature * mask                              # 框外
```

**两条路径**：
- `F_s` = **框内** 4×4 RoIAlign（功能区域）
- `F_e` = 全图去掉框后 **框外** 4×4 RoIAlign（上下文）

**维度表**：

| 变量 | 形状 | 来源 |
|---|---|---|
| `F_I` | [B, 768, 16, 16] | DINOv2 ViT-B/14 特征图（可 1×1 conv 降到 512） |
| `F_s` | [B, 512, 4, 4] → [B, 256, 512] | 框内 RoIAlign |
| `F_e` | [B, 512, 4, 4] → [B, 256, 512] | 框外 RoIAlign |
| `l3` | [B, 512, 64] | PointNet++ sa3（`N_p=64`） |
| `F_j` | **[B, 320, 512]** | `JRA(F_i, l3)`，`N_p+N_i = 64+256` |
| `affordance` | [B, 320, 512] | `ARM(F_j, F_s, F_e)` |
| `up_sample` | [B, 2048, 512] | fp3→fp2→fp1 三级上采样 |
| `F_pa_pool` / `F_ia_pool` | [B, 512, 1] | AdaptiveAvgPool1d |
| `_3daffordance` | [B, 2048, 1] | `out_head(up_sample * F_pa_pool)` |

**`Cross_Attention` 的交互**：
```python
query = proj_q(F_j)                     # [B, 320, 512]
s_key/s_value = proj_s(F_s)             # [B, 256, 512]
e_key/e_value = proj_e(F_e)
I_1 = LayerNorm(F_j + softmax(query @ s_key.T) @ s_value)   # 框内注意力
I_2 = LayerNorm(F_j + softmax(query @ e_key.T) @ e_value)   # 框外注意力
affordance = Conv1d(concat(I_1, I_2))   # [B, 320, 512]
```

**架构本身已经内置了"框内 vs 框外"的差分**（两条独立的 cross-attention），所以设计"差分查询"时不需要再引入新的对比机制，只要在 `ARM` 的输出上加约束即可。

### 2.18 冻结双编码器的可塑性下限（seed1）

**设置**：从 IAGNet 的 `base/best.pt` 开始，冻结 DINOv2（`img_encoder`）和 PointNet++（`point_encoder`，sa1~sa3），只训 JRA + ARM + Decoder（fp1~fp3 + out_head）。**可训练参数 8,043,265 / 95,791,468（8.4%）**，91.6% 被冻结。

**Stage 1（grasp/wrapgrasp，从 base 直接学）**：

| 动作/配对 | 原版 s1 | 冻结 s1 | 保留率 |
|---|---:|---:|---:|
| grasp | 14.40 | **14.26** | **99%** |
| wrapgrasp | 0.96 | **3.23** | **337%** |
| cut | 1.63 | 10.98 | 674% |
| pull | 14.73 | 21.10 | 143% |
| **Knife-grasp** | 10.84 | **11.54** | **107%** |
| **Mug-grasp（3 query 平均）** | 41.97 | **45.75** | **109%** |
| Bowl-wrapgrasp | 0.30 | 7.32 | 24× |
| Vase-wrapgrasp | 0.40 | 10.28 | 26× |
| Baseballbat-grasp | 1.46 / 0.73 | 1.46 / 0.73 | 100% |

**Stage 3（cut/stab，从 base 直接学）**：

| 动作 | 原版 s3 | 冻结 s3 | 保留率 |
|---|---:|---:|---:|
| **stab** | 56.91 | **60.83** | **107%** |
| cut | 15.56 | 12.52 | 80% |
| grasp | 1.21 | 1.53 | 126% |
| wrapgrasp | 0.87 | 1.27 | 146% |
| open | 25.72 | 33.38 | 130% |
| pull | 5.06 | 5.15 | 102% |

`Knife-stab` 逐对（15 个 query）：`67.30→64.50`、`78.52→78.07`、`43.11→52.94`、`45.43→59.02`、`48.25→61.98`、`49.90→66.20`、`66.70→73.36`…… 仅 1 个明显下降（`53.35→32.24`）。

**结论：可以永久锁死双编码器。**
- 三个关键配对全部 ≥ 107%（Mug-grasp 109%、Knife-grasp 107%、Knife-stab 107%）
- `wrapgrasp` 从 0.96 提升到 3.23（3.4 倍）—— 原版全量微调时 encoder 漂移反而在损害它
- s1 的结果两次独立运行完全一致（14.26 / 3.23 / 10.98 / 21.10），可复现

**保留意见**：单 seed；s3-only 是"从 base 直接学 cut/stab"，跳过了 s1/s2，起点与原版 s3 不同。写论文前建议补 3 个 seed，并跑一次完整的 s1→s2→s3 冻结版（脚本已修，`--stages 1,2,3`）。

### 2.19 START 的防覆盖机制与 3DIR 迁移性（立论依据）

原始论文：*START: Structural-semantic collaborative perception transfer for incremental 3D affordance grounding*（DOI: 10.1016/j.eswa.2026.133773）。仓库：`START_PointNet2`，实现报告：`reports/IMPLEMENTATION.md`。

#### 协议与输出形式

- 数据：**PIAD/Seen**，8+3+3+3 划分（不是 3D_AffordanceNet 的 11+4+4+4，也不是 3DIR 的 9+2+2+2+2）
- 输入：**点云 + 显式动作名**（测试时用真实动作名选择输出通道），**无 RGB / 功能框**
- 输出：**多通道**（每个动作一个独立 sigmoid 通道，`Conv1d(192,128,1)→BN→Conv1d(128,1,1)`）
- 训练时逐记录从输出中选取**唯一授权动作的 logits**（`idx`），其余通道不参与该记录的监督

#### 三个损失

```
L = L_supervise + L_SAAT + 5 * L_SCOT
```

| 组件 | 作用 | 依赖 |
|---|---|---|
| `L_supervise` | Focal(γ=2, α=0.25) + Dice | 当前动作的通道索引 `idx` |
| `L_SAAT` | 结构感知注意力转移（teacher-student 的全局/局部上下文 MSE） | **多通道 `old_logits`**（[N, old_count]） |
| `L_SCOT` | 统计与文本语义共现最优传输（Sinkhorn） | **多动作共现统计 + CLIP 文本语义矩阵** |

#### 防覆盖的核心：SCOT

```python
# 原文式 (6)：同阶段同时可见的多动作标签 → 共现统计
# 缺失的动作对 → CLIP ViT-B/32 文本语义相似度补全
cost, prior_audit = build_cost(statistics, 'weights/semantic_prior.pt',
                               semantic_threshold, prior_energy)
...
a = old_logits.softmax(-1)      # [N, old_count]  旧教师的预测分布
b = old_student.softmax(-1)     # [N, old_count]  当前学生的预测分布
scot = sinkhorn_value(a, b, cost, epsilon, iterations, tolerance)
```

**SCOT 的语义**：用动作间的语义 cost 做最优传输，约束"旧动作的预测分布"不被新动作的梯度改写。这是"共现物体部件覆盖"的正则。

#### 迁移到 3DIR 的三个硬失效点

| START 的依赖 | 3DIR 的实际情况 | 后果 |
|---|---|---|
| **多通道输出**（[N, old_count] 的 softmax） | **单通道**（[N, 1]，每 query 一个动作） | `sinkhorn_value(a, b, cost)` 的边际分布不存在 |
| **显式动作名**（CLIP text → cost） | **无动作名**，只有图像 | `build_cost` 无法构建 |
| **同几何的多动作共现统计** | **一 query 一动作**，没有共现 | 实测共现矩阵为空 |

**所以 START 的防覆盖机制在 3DIR 上必然失效。** 这不是实现问题，是条件类型不匹配：START 的"动作间关系"来自**文本语义**，3DIR 只有**当前动作的视觉特征**。

#### 立论：为什么需要「跨条件历史视觉特征采样伪标签」

START 用**文本语义**定义动作间关系，从而知道"哪些旧动作区域容易被新动作覆盖"。3DIR 没有文本，但它有**历史动作的视觉分布**（每个旧动作在支持图像上的视觉特征统计）。这条路径可以补上同样的信息：

1. **用历史动作的图像特征统计**（均值/协方差，几 KB）替代 CLIP 语义矩阵，构建"动作间视觉关系"；
2. **配合当前点云**生成旧部件的伪标签（在历史视觉方向上激活的点）；
3. **避免与当前新图像条件冲突**：伪标签只作用于旧动作通道/旧属性，不进入新动作的读出。

这正好绕开 START 的三个失效点：不需要多通道 softmax、不需要动作名、不需要共现统计。

**一个已知的额外约束**：3DIR 有 8 个 Base 独有物体（2.16）在整个增量过程中从未出现，它们的旧部件伪标签只能来自**历史视觉统计 + 当前点云**，没有任何增量数据能提供监督。这一点必须在设计里显式处理（例如只对"出现过的物体"生成伪标签，其余靠冻结 Base 输出路径保护）。

---

## 3. 全监督上界（23 类联合训练）

### 3.1 设置

- **full**：23 类 × 16082 个训练实例（不截断），DGCNN，warm 8 + formal 8 epoch，batch 16
- **partial**：23 类 × 16082 个训练视角（从 64328 个官方视角中每个物体取一个），配置 `dgcnn_partial.py`
- **PointNet++ 未跑**：仓库里的 `_ext.cpython-310-x86_64-linux-gnu.so` 是给原项目环境（torch 2.8.0+cu128）编译的，在可用的 torch 2.5.0+cu124 环境下导入失败（`ModuleNotFoundError: Could not import _ext`）。重新编译需要写 `/nas_data/ljy`，违反只读约束。
- 验证集：full 2285 个实例 / partial 9140 个视角
- **评估口径**：58 个有效对的 per-pair macro（与 APGraph 的 38.60 完全同口径），不是 `utils.evaluation` 的 18 动作平均

### 3.2 结果

| 模型 | All mAP | All maIOU | All mAUC | Base 28 对 mAP | Novel 30 对 mAP |
|---|---:|---:|---:|---:|---:|
| 11 类 base（full，增量前） | 35.96 | 17.84 | 78.24 | **43.87** | 28.57 |
| APGraph 增量后（full） | 38.60 | 20.02 | 82.52 | **43.87**（未变） | 33.76 |
| **全监督 23 类（full）** | **44.60** | **24.94** | **86.58** | 41.64 | **47.37** |
| APGraph 增量后（partial） | 32.80 | 16.55 | 77.16 | 待填 | 27.49 |
| **全监督 23 类（partial）** | **34.42** | **17.32** | **76.98** | 34.81 | 34.05 |

全监督的训练曲线（`utils.evaluation` 口径，18 动作平均）：
- full：warmup 4 = 40.58%、warmup 8 = 42.22%、formal 4 = 42.16%、formal 8 = **42.90%**（58 对口径 44.60）
- partial：formal 8 = **35.38%**（58 对口径 34.42）

full 最后 4 个 epoch 只涨 0.74pp，基本收敛（但仍有轻微上升，上界可能被低估约 0.5–1pp）。

### 3.3 分解：full 上损失 100% 来自「少样本」，partial 上几乎已经没有损失

**APGraph 的增量增益（相对 11 类 base，full）**：mAP +2.64、maIOU +2.18、mAUC +4.28 —— 全部来自 Novel 30 对（28.57 → 33.76），Base 28 对完全不变。

**APGraph 相对全监督的差**：

| 协议 | 全监督 All mAP | APGraph All mAP | 差 | 全监督 maIOU | APGraph maIOU | 差 |
|---|---:|---:|---:|---:|---:|---:|
| full | 44.60 | 38.60 | **−6.00** | 24.94 | 20.02 | **−4.92** |
| partial | 34.42 | 32.80 | **−1.62** | 17.32 | 16.55 | −0.77 |

**full 的逐组分解**：

| 范围 | 全监督 | APGraph | 差（APGraph − 全监督） |
|---|---:|---:|---:|
| Base 28 对 mAP | 41.64 | 43.87 | **+2.23**（APGraph 更好） |
| Novel 30 对 mAP | 47.37 | 33.76 | **−13.61** |

**四条结论：**

1. **full 上「增量顺序」没有造成损失。** Base 28 对上 APGraph（43.87）比全监督（41.64）还高 2.23pp。原因是全监督的 23 类联合训练被数据不平衡稀释了：base 类里 Bag 只有 88 个样本、Bed 127、Bowl 132，而 novel 类里 Table 有 5593、StorageFurniture 1531、Vase 735。联合训练时 novel 类主导了梯度，base 类反而被牺牲。
2. **full 上全部 6.00pp 的 mAP 差距来自 Novel 30 对**：全监督 47.37 vs APGraph 33.76，差 13.61pp。这是「每类 5 个支持对象」与「每类 110–5593 个训练样本」的差距，是 few-shot 的固有代价。
3. **partial 上 few-shot 损失只剩 1.62pp，但全监督本身掉了 10.18pp**（44.60 → 34.42）。也就是说：**单视角缺失比 5-shot 更根本**。「每个物体只有一个视角」这个限制，连全监督都扛不住；在这个条件下，5-shot 增量（32.80）已经非常接近全监督上限（34.42）。
4. **骨干极限被低估了一点点**：full 的 formal 最后 4 个 epoch 还在涨（42.16 → 42.90），如果训到 30–40 epoch，58 对口径可能到 46–47。但即使按当前 44.60 算，"少样本损失 ≫ 增量损失"的结论不会变。

**对换方向的含义**：
- **full 协议**：继续优化"增量策略"的收益上限是 0（Base 已经不损失），真正的空间在"如何用 5 个支持对象学到接近全监督的新类"——那是 13.61pp。
- **partial 协议**：连全监督都只有 34.42，**视角补全/多视角融合**才是更大的杠杆（10.18pp），而不是 few-shot 或增量。

---

## 4. 判断

### 4.1 3DIR 上真正未饱和的空间在哪

1. **Novel 动作的局部区域（grasp / wrapgrasp / stab / move）**：APGraph 这四项 4.1–11.6，是所有方法共同的坑。而且 `Baseballbat-grasp`、`Scissors-grasp`、`Knife-grasp` 的 AUC 只有 31–47，连排序都没学会。
2. **区域面积/边界**：`Knife-stab` AUC 96.8 → aIOU 11.3。排序已经接近完美，差的是"标多大"。
3. **概率饱和**：APGraph 的输出在整片功能件上饱和到 1.000（p90=1.0），导致 20 档阈值扫描没有任何工作点。这不是校准问题（在 3D_AffordanceNet 上 perclass Oracle 只有 +2.0pp，partial 上 +1.2pp），而是**输出分布退化成二值**。
4. **图像方法的稳定性**：它们能在 novel 动作上做到 18，说明跨模态信息对局部功能件是有价值的；问题是 600 步增量就把 Base 打崩。任何"冻结 Base + 只学新头"的做法在这里都有 27pp 的空间。

### 4.2 不建议的方向

- **继续做"无效配对抑制"**：Oracle A 在 full 和 partial 上都严格为 0。
- **继续做"阈值/校准"**：global 1–2pp，perclass 2pp，且方向不可迁移。
- **做输出尺度校准（温度缩放 / 偏置平移 / LayerNorm）**：在 Base 9 上选到的最优 `(c, T)` 就是 `(0, 1)`，Novel 8 在任何非平凡变换下都变差。aIOU 是阈值平均，对单调变换近似不变（见 2.13）。
- **把 3DIR 的低分归因于"图像质量/遮挡"**：stage0 的 87 AUC 已经否定了这一点。
- **针对 Knife-stab 做阈值扫描**：最优单阈值 IoU 只有 0.071，20 档里最好的也只有 0.071。
- **把 APGraph 的新动作差归因于防遗忘门**：关掉门控后 novel 只涨 0.38pp，base 崩 11pp。

### 4.3 对后续方法设计的直接含义

1. **Base/Novel 的读出要解耦，而不是做输出尺度校准。** Base 属性的响应通过 `z` 的求和给 Novel 动作注水（zero_old 掉 5.23pp），而 `move`/`grasp`/`stab` 关掉 Base 后反而更好。可行的方向是：Novel 动作只用 Novel 属性读出（或对 Base 的贡献做面积归一化），同时保留 Base 动作的冻结输出。**注意不要走"加 LayerNorm 修饱和"这条路**——见 4.2，校准增益是 0（aIOU 对单调变换不变）。
2. **局部握持区需要专门的几何归纳偏置**：`grasp`/`wrapgrasp` 的 GT 是手柄/把手环/瓶身一小段。纯点云属性图在 `Mug-grasp` 上只有 4.30，而带图像的 IAGNet 有 41.19。凹形包围关系（曲率符号、内腔、闭合环）是点云可以算但 APGraph 没显式建模的量。
3. **增量保护要按配对，不是按动作**：IAGNet 的 `grasp` 动作级 aIOU 从 stage1 的 14.40 涨到 stage4 的 27.27，但 `Knife-grasp` 这个配对从 10.84 掉到 0.10。动作级平均把配对级遗忘完全掩盖了。任何"按动作统计"的保护机制都会漏掉这种破坏。
4. **"全量微调 + 只训当前 session"是 IAGNet 的遗忘来源，也是它能学到新动作的原因。** 如果要两者的优点，需要"部分解冻 + 按配对的重放/正则"，而不是纯冻结或纯微调。
5. **不要指望"更强的增量策略"解决 3DIR**：APGraph 的增量已经是零遗忘（Base 43.32 恒定），它的 30.72 是"Base 满分 + Novel 16.5"。要把 All 从 30 推到 40，必须让 Novel 从 16.5 推到 40，这需要新的几何表征，不是新的增量策略。

---

## 附：产物

| 路径 | 内容 |
|---|---|
| `/data01/jh/afford_diag/oracle_full_k5.json` | full 的 raw/oracleA/oracleB/global/perclass |
| `/data01/jh/afford_diag/oracle_partial_k5.json` | partial 同上 |
| `/data01/jh/afford_diag/threedir_breakdown.json` | 3DIR 逐对/逐动作/逐物体/交叉表 |
| `/data01/jh/afford_diag/threedir_by_action.csv` | 17 动作 × 5 方法的 aIOU/AUC |
| `/data01/jh/afford_diag/threedir_by_object.csv` | 21 物体 × 3 方法的 aIOU |
| `/data01/jh/afford_diag/base11_on23.json` | 11 类 base 在 58 对口径的结果 |
| `/data01/jh/afford_diag/apgraph_gate_probe.json` | raw / gate_off / zero_new 三组消融 |
| `/data01/jh/afford_diag/afford_saturation.json` | 3D_AffordanceNet 58 对的饱和统计 |
| `/data01/jh/afford_diag/joint23/` | full 全监督 manifest + base + `joint_metrics.json`（58 对评估） |
| `/data01/jh/afford_diag/joint23_partial/` | partial 全监督 manifest + base（16082 views / 9140 val） |

脚本（`afford_scripts/`）：`afford_oracle.py`、`threedir_breakdown.py`、`threedir_point_dissect.py`、`threedir_image_confound.py`、`threedir_stage_trace.py`、`apgraph_gate_probe.py`、`apgraph_saturation_probe.py`、`afford_saturation.py`、`make_all23_manifest.py`、`make_all23_manifest_partial.py`、`joint_eval.py`、`joint_compare.py`
