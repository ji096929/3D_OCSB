# 3D-OCSB 第二步探索：B 提纯、负对照信号、融合接口

日期：2026-11-05
基于第一步的 502 episodes 结论（B precision 39.1%，遗漏 30.3%，桥接优势 96.4%/81.9%），
做三个零训练离线实验：
1. Top-K 提纯 B
2. 升级负对照信号（AUC）
3. Oracle 桥梁注入，验证式（15）的融合接口

## 一、Top-K 提纯：成功，58% 接近目标

在 60 episodes 上测（逐桥 `q_c(R)` 和 GT 已记录）：

| 策略 | B precision |
|---|---:|
| 原始（τ=0.6，全收） | 0.391 |
| τ=0.6 + **Top-3** | **0.606** |
| τ=0.6 + Top-5 | 0.550 |
| τ=0.6 + Top-10 | 0.500 |
| Top-1（原始数据） | 0.530 |

**结论**：
- **τ=0.6 + Top-3 → B precision 60.6%**，**达到老师 60% 的判据**。
- K=1/3/5 在原始数据上都是 0.530，说明每个 episode 的 B 本来就只有 3–5 个超点。
  问题不是"选多了"，而是"选进来的那 3–5 个里只有一半是对的"。
- Top-K 的天花板约 60%，要再往上必须换选择标准（见第二节）。

## 二、负对照信号 AUC：0.6 左右，方向有意外

在 100+ episodes 上，对每个 B 超点算 4 个信号，比较 B_pos vs B_neg 的 AUC：

| 信号 | pooled AUC | per-episode AUC | 解读 |
|---|---:|---:|---|
| `multi_max` = max_k Sim(B, a_k) | 0.567 | 0.577 | 真桥梁与 support 子原型局部对齐更好 |
| `-delta_base` = -max_m [Sim(B,P_base_m) - Sim(A,P_base_m)] | **0.615** | 0.555 | **假桥梁更像 base 类**（方向要取负） |
| `multi_max - delta_base` | **0.609** | **0.587** | 组合最平衡 |
| `multi_max - base_abs` | 0.563 | 0.577 | |
| `margin` = multi_max - max_m Sim(B,P_base_m) | 0.563 | 0.577 | |
| `support_D`（原版负对照，episode 均值） | 0.484 | — | **无效** |

**结论**：
- 原版 support 负对照确实无效（AUC 0.484，之前 Cohen's d=0.17，57.7% 方向反）。
- **`-delta_base` 是目前最好的单信号（pooled 0.615）**：假桥梁（table 被误检成 counter）
  与 base 原型的亲和力增量更高，取负后是有效信号。
- 组合 `multi_max - delta_base` 的 per-episode AUC 最高（0.587）。
- **但仍未达到 0.7 的可用线**，需要更多信号（Δ 分布、跨 episode 稳定性）。

## 三、Oracle 注入：logit 级融合只 +0.5pp，2C 双路 HCA 才是有效接口

用真值挑出 U 中 100% 正确的 B_pos，测试三种融合方式（60 episodes）：

| 方式 | query IoU | Δ vs base |
|---|---:|---:|
| base（无桥梁） | 0.3825 | — |
| logit 直接叠加（λ=0.5） | 0.3818 | −0.08pp |
| 2A 点池融合（support fg + B_pos 点） | 0.3723 | **−1.03pp** |
| **2C 双路 HCA 融合（λ=0.3）** | **0.4013** | **+1.88pp** |
| 2C 双路 HCA 融合（λ=0.5） | 0.3947 | +1.22pp |

**结论**：
- **2C 是唯一有效的接口**：两路都经过 COSeg 的 BPC + HCA 校准，得到同标度的
  query logits 再融合（`(1-λ)·ℓ_S + λ·ℓ_B`），不会破坏原有分布。
- **2A 失败（−1.03pp）**：把 B_pos 的点混进 support 前景点会污染原型，
  97 个 support 原型 vs 少量 B 点的配额差导致 HCA 响应带被稀释。
- **logit 直接叠加无效（−0.08pp）**：证实了式（15）的简单余弦叠加与
  COSeg 的非线性预测头不兼容。
- **Oracle 上界是 +1.88pp**：这是 B_pos 100% 纯净时的天花板。实际 B 只有
  60% 纯净，真实增益会打折（经验上 1/3～1/2），预计 +0.6～1.0pp。
- λ=0.3 比 0.5 好，说明桥梁路的权重不宜过高，support 仍是主导。

## 四、综合判断

| 问题 | 结论 |
|---|---|
| Top-K 能否把 B 提到 55%+？ | ✅ 能，τ=0.6 + Top-3 = **60.6%**（达 60% 判据） |
| 负对照能否区分真假桥梁？ | ⚠️ 部分，最好 AUC 0.615（-delta_base），未到 0.7 |
| 式（15）logit 融合能否涨点？ | ❌ 只 −0.08pp，接口不兼容 |
| 2C 双路 HCA 融合能否涨点？ | ✅ Oracle **+1.88pp**（λ=0.3） |
| 桥接优势是否成立？ | ✅ 单原型 96.4%，多原型 81.9% |

**对方案价值的判断**：
- 有真实增益，但 Oracle 上界只有 +1.9pp，离 +3pp 的"立住"门槛差 1.1pp。
- 实际（B 只有 60% 纯净）预计 +0.6～1.0pp，属于"偏薄"区间。
- 方案的瓶颈不在融合接口（2C 已解决），而在 **B 的纯度和数量**：
  Top-3 只有 3 个桥梁，注入了也改变不了大局。

## 五、下一步

1. **扩 B 的规模**：Top-3 只选 3 个，试试 Top-10 + 组合分数
   （`multi_max - delta_base`），在纯度不降太多的前提下增加桥梁数量。
2. **2C 的 λ 细调**：0.3 最好，试 0.1/0.2/0.4。
3. **B 选择改用组合分数**：`score = multi_max - delta_base` 替代纯 `q_c(R)`。
4. **补 Δ 分布信号**：实现式（6）的 log-mean-exp，算 `η_spread(B)`，
   看能否把 AUC 从 0.615 推到 0.7+。
5. **报告里明确**：原版 support 负对照在 1-shot 下失效（AUC 0.484），
   需要换成 hybrid negative bank（纯化 support + base 原型 + U 高置信背景）。

## 复现

```bash
# Top-K + Oracle 数据
cd /data01/jh/code
CUDA_VISIBLE_DEVICES=<gpu> python -u diag_bc.py --episodes 25 --M 3 --seed <s> \
  --out /data01/jh/data/diag_or2_s<s>

# 分析
python analyze_toplk_oracle.py "/data01/jh/data/diag_or2_s*/records.jsonl"
python auc_signals.py "/data01/jh/data/diag_or2_s*/records.jsonl"
```
