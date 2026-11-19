# 整场景口径评估报告

日期：2026-11-17
状态：**负结果 + 一处口径错误的更正**

---

## 0. 结论速览（含一处口径更正）

### 0.1 必须先更正：论文口径是 1m block，不是整场景

论文第 6 页 Implementation Details 原文：

> We employ the data processing strategy from [18, 56]. **The 3D scene is divided
> into 1m × 1m blocks to increase data samples**, and raw input points are
> grid-sampled with a grid size of 0.02m. After voxelization, if the input point
> count exceeds 20,480, we randomly sample 20,480 points to control the input
> size. ... **During testing, we sample 1,000 episodes per class** in the 1-way
> setting and 100 episodes for each combination in the 2-way setting.

代码侧也一致：`Scannetv2_FS`（训练）与 `Scannetv2_FS_TEST`（测试）共用同一个
`data_root`，四处加载点云全是 `np.load(os.path.join(self.data_root, name + ".npy"))`，
**没有任何切块逻辑**——`data_root` 里放 block 就是 block 口径，放整 scene 就是整 scene 口径。
官方 yaml 的 `data_root` 留空（`# Fill in the pre-processed data path`）。

**所以 Table 2 的 ScanNet 数字（fold-0 `sc0_1w1s` 对应 41.73）是 block 口径。**
本报告第 1–7 节的整场景实验**不是论文口径**，它只用来回答
「目标占比极低时方法是否还有效」。

### 0.2 正确的对比：block 口径 38.70% vs 论文 41.73%

按官方 `get_class2scans` 的规则（目标类点数 > max(scene 总点数 5%, 100)）
过滤 539 集后重算：

| query 目标占比阈值 | 集数 | Baseline Cumul | 自桥接 | Δ |
|---|---:|---:|---:|---:|
| 不过滤 | 539 | 0.3229 | 0.3293 | +0.64 |
| ≥1% | 502 | 0.3411 | 0.3466 | +0.55 |
| ≥3% | 469 | 0.3575 | 0.3644 | +0.69 |
| **≥5%（官方阈值）** | **435** | **0.3870** | **0.3953** | **+0.83** |
| ≥10% | 371 | 0.4267 | 0.4333 | +0.66 |
| ≥20% | 258 | 0.5152 | 0.5205 | +0.52 |

539 集 query 的目标占比分布：p25 = 7.9%、p50 = 19.4%，**gt < 5% 的占 18.8%**。

结论：

1. 32.29% → 41.73% 的 9.44pp 差距中，**约 6.4pp 来自候选池没有 5% 过滤**；
   剩下 3.0pp 可由权重重训版（README 自述 "retrained by us"）与抽样差异解释。
2. **我们的 block 口径基准是对的**，不是复刻错误。
3. **方法增益在官方口径下仍然成立**（435 集上 +0.83 Cumul），
   但相对 41.73 只有 2.0% 相对提升，**不足以支撑一篇新论文的卖点**。

### 0.2.1 官方独立验证：38.16%

用官方 `main_fs.py` + 官方权重，把 `data_root` 换成 `blocks_bs1_s1/data`
（官方会自动按 5% 阈值生成 `class2scans.pkl`，我们已核对池子规模：
chair 1829、cabinet 680、floor 6017 等），跑 200 集（每类 20）：

| 类 | bathtub | bed | bookshelf | cabinet | chair | counter | curtain | desk | door | floor | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 官方 block | 0.2355 | 0.4877 | 0.5398 | 0.3294 | 0.4899 | 0.1638 | 0.2742 | 0.3964 | 0.3262 | 0.5733 | **0.3816** |

与我们用 5% 过滤重算的 **38.70%** 只差 0.5pp（抽样差异），**两条完全独立的路径互相印证**。
与论文 41.73 差 3.6pp。

### 0.3 整场景实验的独立价值

即便不是论文口径，整场景实验回答了一个论文回避的问题：**目标占比极低时会发生什么**。

- 整场景 baseline 20.9%，Oracle-B+C 上限仅 +0.11pp（block 口径 +4.90pp）。
- 根因：整场景的失败模式是**过预测**（FP/TP 达 4.98–22.1），不是漏检（FN 仅 7%–17%）。
  OCSB 修的是漏检，方向不对。
- 唯一有效的干预是**前景比例校准**：按固定 10% 分位切，0.2298 → 0.2882（+5.8pp）；
  但 oracle 占比校准也只有 0.3063，说明排序质量本身也差。

**这与官方 5% 阈值的设计是一致的**：官方过滤掉的正是模型失效的区间，
OCSB 在那个区间也失效。两条证据互相印证，而不是矛盾。

### 0.4 撤回上一版的一个错误结论

上一版写过「论文 41.78 用官方发布权重无法复现」。**这句话作废**——
它建立在「论文用整场景」这个错误前提上。整场景 24.3% 的测量本身没错
（官方代码在 `scenes/data` 上确实是这个数），但**不能用来判定论文不可复现**。

---

## 1. 整场景口径的官方测量：24.3%（非论文口径）

**先强调：这不是论文口径**（论文是 1m block，见 0.1）。这组数字只说明
"官方代码喂整场景会给出什么"，用途是支撑第 3 节的诊断。

用官方 `main_fs.py`、官方权重 `sc0_1w1s.pth`、官方数据处理 `data_prepare_v101`、
官方 episode 缓存（每类 20 个，共 200 个 episode）：

| 类 | bathtub | bed | bookshelf | cabinet | chair | counter | curtain | desk | door | floor | **mIoU** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IoU | 0.2534 | 0.2099 | 0.1995 | 0.1790 | 0.1801 | 0.1639 | 0.1841 | 0.1624 | 0.1209 | 0.7785 | **0.2432** |

论文 Table 2 的 block 口径是 **41.73**（fold-0）。整场景数字与它不可比，
所以「差 17.4pp」这个说法没有意义，不要引用。

官方 README 在 Model weights 一节有一句原话：

> Please note that these weights have been **retrained by us**, which may have slight differences from reported results.

这解释了为什么 0.2 节里过滤后仍差 3.0pp——但 3.0pp 属于可接受范围，
不是"复现不出来"。

---

## 2. 我们的复刻是对的

此前整场景 50 个任务得到 Cumul 0.2089 / Arith 0.2230。为了确认这不是我们实现的 bug，
逐层做了核对：

| 检查项 | 结果 |
|---|---|
| 类别编码 | 官方打印 `Classes: [10 6 7 19 3 13 12 5 14 2]`，与我们的 `NEW_IDS` 完全一致 |
| 前向结构 | `lin1 = Linear(100→192)`；`class_reduce = LayerNorm(192) → Conv1d(2,1,1) → ReLU`，归约的是 `n_way+1` 维；`sparse_embeddings = cat([bg, fg])` 顺序一致 |
| 超参 | `n_subprototypes=100`、`voxel_size=0.02`、`voxel_max=20480`、`n_way=1`、`k_shot=1` |
| base 引导 | 官方 `validate` 传 `epoch=5`，走 `base_guidance` 分支；我们同样传 `base_prototypes` |
| 数据处理 | 同一函数 `data_prepare_v101`，缓存目录名 `..._pts_20480_vs_0.02_rdmsp` 印证随机采样 |

**直接对拍**（同一个官方缓存 episode，bathtub）：

```
support 前景 1614 点，query 前景 1269 点 (6.20%)
官方 forward    iou 0.1463  tp 1234  fp 7168  fn 35   pred_fg 41.03%
```

**官方自己就把 query 的 41% 预测成前景**，而真值只有 6.2%。所以此前观察到的
"FP 是 TP 的 5–22 倍"是**官方模型行为**，不是我们复刻引入的。

---

## 3. 整场景下 OCSB 的上限：+0.11pp

50 个任务（每类 5 个），官方池（见第 5 节），λ=0.3：

| 配置 | Cumul | Arith | ΔCumul |
|---|---:|---:|---:|
| Baseline | 0.2089 | 0.2230 | – |
| Naive-B（U 中 p0>0.7） | 0.2025 | 0.2126 | −0.64 |
| Oracle-B（p0>0.7 且真值为前景） | 0.2095 | 0.2183 | +0.06 |
| Oracle-B+C（U 中全部真前景） | 0.2100 | 0.2201 | +0.11 |

对照 block 口径（539 集）：Oracle-B +4.34、Oracle-B+C +4.90。

**差距两个数量级。**

### 为什么失效

逐类统计（Baseline）：

```
class 10 : iou 0.163  tp  6797  fp 33864  fn  1029
class 14 : iou 0.043  tp  2761  fp 61134  fn   219
class 2  : iou 0.773  tp 20467  fp  1674  fn  4337
```

**FN/(TP+FN) 只有 7%–17%，FP/TP 是 4.98 / 22.1 / 0.08（floor 是唯一不过预测的类，因为它占了大半场景）。**

OCSB 的全部机制都建立在"模型漏检了一部分真前景（FN），把它们补进原型就能提升"这个假设上。
整场景里 FN 本来就快没了，瓶颈是**误检**。修一个几乎不存在的问题，自然没有增益。

这就是 Oracle 也归零的原因——**即便直接给全部真前景，也没有东西可修**。

---

## 4. 采样策略决定绝对水平，但都到不了 0.42

同一批 episode、同样 20 集，只换 `voxel_max` 的采样方式：

| 采样 | Cumul | Arith | 平均 query 前景占比 | 平均点数 |
|---|---:|---:|---:|---:|
| `v101` 随机采样（官方当前代码） | 0.2094 | 0.2298 | 11.1% | 20480 |
| `sphere` 球状裁剪 + 中心化（旧版 `data_prepare`） | 0.2921 | 0.2841 | 15.3% | 20480 |
| `none` 不采样（保留 voxel 后全部点） | 0.2029 | 0.2289 | ~10% | 89722 |

数据：`/data01/jh/data/ocsb_cache/sampler_probe_*.json`（各 20 集，同一批 episode）

**关键观察**：

1. **随机采样不改变前景占比**。`v101` 与 `none` 的 query 前景占比都是约 10%，
   因为前景和背景被同比例采样。而且两者 Cumul 只差 0.7pp（0.2094 vs 0.2029）——
   **「随机采样把目标稀释了」这个假设被彻底否定**，保留 9 万点也不比 20480 点更好。
2. **球状裁剪抬高了前景占比**。它只保留场景中心 20480 个点，而目标常在中心附近，
   于是占比从 11% 升到 15%。**这是采样偏差，不是模型更强。**
3. 球状裁剪会丢目标（出现了 `q_fg = 0` 的 episode），也会引发崩溃
   （出现 `pred_fg = 99.88%` 的 episode）。

**三种采样没有一种能到 0.42。**

---

## 5. 候选池的 5% 过滤：9.44pp 的主要来源

官方 `get_class2scans` 有一道过滤，scene 级和 block 级用的是**同一段代码**：

```python
min_ratio = 0.05
min_pts = 100
threshold = max(int(data.shape[0] * min_ratio), min_pts)
if num_points > threshold:
    class2scans[class_id].append(scan_name)
```

即**只有目标类点数超过该文件总点数 5% 的样本才进入候选池**。

### 5.1 整场景口径（scene 级）

我们原来的 `scene_classes.json` 是「只要出现该类就算」。两者对比：

| 类 | 出现即计入 | 官方 5% 阈值 | 原先 IoU |
|---|---:|---:|---:|
| counter (13) | 47 | **15** | 0.061 |
| door (14) | 219 | **95** | 0.043 |
| desk (5) | 84 | **31** | 0.122 |
| bathtub (10) | 31 | **15** | 0.163 |
| floor (2) | 312 | 311 | 0.773 |

低 IoU 的四个类正好是缩水最狠的四个。但用官方池重跑 20 集，Baseline 仍是 0.2094
——**在整场景口径下池子不是主因**，主因是模型在整场景输入上的水平（第 1 节）。

### 5.2 block 口径（论文口径）—— 这里才是关键

`make_manifest.py` 里是 `c in b['classes']`，同样没有 5% 过滤。
539 集 query 的目标占比分布：

```
p1  = 0.086%   p5  = 0.587%   p10 = 1.835%   p25 = 7.899%
p50 = 19.413%  p75 = 37.380%  p90 = 63.587%  p99 = 98.628%
gt < 1%  占  6.6%
gt < 5%  占 18.8%
gt < 10% 占 30.4%
```

**18.8% 的 episode 目标占比不足 5%**，官方会全部过滤掉。
按 5% 过滤后 baseline 从 0.3229 升到 0.3870（见 0.2 节表）。

### 5.3 更正上一版报告的一句错话

上一版写：「1m block 里出现一把椅子，椅子占比天然就高，所以 block 口径的
0.3229 隐含了一个比官方 5% 阈值更强的目标占比偏差。」

**方向说反了。** 事实是：block 池没有过滤，**包含了 18.8% 的极低占比 episode，
把平均值拉低**，不是抬高。过滤后 0.3870 才接近论文的 41.73。

---

## 6. 由此确认的机制：模型的前景预测比例不自适应

同两个 bathtub episode，只换采样：

| 采样 | query 前景占比 | 模型预测前景占比 | IoU |
|---|---:|---:|---:|
| 随机采样 | 10.1% | 21.2% | 0.430 |
| 随机采样 | 10.0% | 24.0% | 0.383 |
| 球状裁剪 | 16.0% | 20.7% | 0.650 |
| 球状裁剪 | 16.0% | 20.9% | 0.645 |

**模型输出的前景比例几乎固定在 20%–24%，不随目标占比变化。**
目标占 16% 时 IoU 0.65，目标占 10% 时 IoU 0.43——多出来的预测全是 FP。

这直接回答了「背景占绝大多数时效果更好还是更差」：**更差，且机制明确**——
模型没有"按目标占比自适应"的能力，背景越多，固定比例的前景预测里 FP 越多。

---

## 7. 比例校准：+4.6pp，但远不够

对 baseline 的概率做后处理，按不同比例取 top-k 分位作为预测（20 集，`v101` 口径）：

| 阈值来源 | 平均 IoU | Δ vs argmax |
|---|---:|---:|
| argmax（原始，p>0.5） | 0.2298 | – |
| 固定 5% 分位 | 0.2105 | −2.0 |
| **固定 10% 分位** | **0.2882** | **+5.8** |
| 固定 20% 分位 | 0.2519 | +2.2 |
| **support 前景占比** | **0.2761** | **+4.6** |
| **query 真值占比（oracle）** | **0.3063** | **+7.7** |

三条结论：

1. **比例校准确实有效**，零成本后处理就能 +5.8pp（固定 10%）。这比 OCSB 在整场景的
   +0.11pp 大 50 倍，是整场景口径下目前唯一被验证有效的干预。
2. **`support` 占比没有额外价值**：0.2761 甚至低于固定 10% 的 0.2882。
   说明 support 与 query 的目标占比相关性不强，估不出来也不必估。
3. **瓶颈不只是比例。** 即使把比例设成 oracle（0.3063），也远未到论文水平。
   所以模型在整场景上的**排序质量本身也差**，比例校准只能吃掉一部分损失。

### 对第 3 节的修正

第 3 节说过「排序几乎是对的，错的是切在哪里」——**这句话只在部分类成立**。
逐集看，bathtub 校准后能到 0.60–0.68，而 cabinet 的一集从 0.134 崩到 0.013。
**排序质量是类相关的**，不能一概而论。

但第 6 节的结论不变：模型输出的前景比例固定在 20%–62%，不随目标占比自适应，
这是低占比场景掉点的直接原因之一。

---

## 8. 对后续工作的影响

1. **block 口径是与论文对话的正确口径**。539 集必须按官方 5% 阈值过滤成
   **435 集**，baseline 38.70% vs 论文 41.73%，差 3.0pp 可接受。
   后续实验都应该切到这个过滤后的子集上，而不是继续用 539 集。
2. **方法增益在官方口径下成立且更大**：435 集上自桥接 **+0.83 Cumul**
   （原 539 集是 +0.64）。过滤掉低质量 episode 后增益反而上升，
   说明方法不是靠那些 episode 撑起来的。
3. **但 +0.83 相对 41.73 只有 2.0% 相对提升，不足以支撑一篇新论文。**
   要投稿必须找到量级更大的改进，而不是在这个方向上继续磨小数点。
4. **整场景口径只用于回答「目标占比极低时方法是否还有效」**。答案是没有，
   因为那里的瓶颈是过预测而非漏检。官方 5% 阈值过滤掉的正是这个区间——
   两条证据互相印证，不是矛盾。
5. **整场景上唯一验证有效的是前景比例校准**（+5.8pp）。如果要把方法扩展到
   低占比场景，应当以它为基线；注意 oracle 校准也只有 0.3063，
   说明还需要同时改善排序质量。

---

## 9. 复现

```bash
# 官方评测（需要 fake_tp 垫片，官方 main_fs.py 会撞 torch_points3d 依赖链）
cd /data01/jh/code/COSeg
CUDA_VISIBLE_DEVICES=2 python /data01/jh/code/run_official_eval.py \
    --config config/scannetv2_COSeg_fs.yaml test True \
    data_root /data01/jh/data/ScanNet/scenes/data \
    weight /data01/jh/weights/sc0_1w1s.pth \
    pretrain_backbone /data01/jh/weights/sc_s0pre \
    cvfold 0 n_way 1 k_shot 1 num_episode_per_comb 20 \
    train_gpu "[0]" multiprocessing_distributed False workers 4

# 官方候选池（5% 阈值）
python /data01/jh/code/make_class2scans.py

# 整场景评估（官方池）
cd /data01/jh/code
python scene_eval.py --pool official --per-class 5 --out .../scene_off.json

# 采样策略探针
python scene_sampler_probe.py --mode {v101,sphere,none}

# 前景比例校准探针
python probe_calib.py
```

产物：
- `official_test.log`：官方 200 集逐类结果
- `scene_off_s*.json`：官方池整场景四路结果
- `sampler_probe_*.json`：三种采样策略（各 20 集）
- `probe_calib.log`：比例校准结果
- `class2scans.pkl`：官方候选池（5% 阈值）
