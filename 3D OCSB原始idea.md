# 3D OCSB实验复盘总结

> **状态（2026-11-19）：本 idea 已结案，两轮验证均为负结果。**
> - 路线 1（COSeg + 外部无标签池 U）：`docs/ocsb_legacy/FINAL_VERDICT.md`
> - 路线 2（GFS-VL / ScanNet200 伪标签改造）：**[`GFS_VL_CONCLUSION.md`](GFS_VL_CONCLUSION.md)**
>
> 一句话：桥梁池（AUC 0.496）与基类负对照（0.754 < 纯原型 0.795）在 1-shot 3D 上都没有可用信号；
> 瓶颈在新类分类器本身，不在伪标签。

> 对话 URL: https://chatgpt.com/c/6aec4db7-abd8-83e8-a04a-aa9b8739b3c4
> 导出时间: 2026-11-12 22:05:30

---

**A：**

# 3D-OCSB：面向半监督小样本点云分割的遗漏条件化选择性语义桥接

论文 Method 中文初稿 · 面向 CVPR / ICCV / NeurIPS 等计算机视觉与机器学习顶会 · 包含学生实现说明

Omission-Conditioned Selective Bridging for Semi-Supervised Few-Shot 3D Point Cloud Segmentation

以下采用正式论文的叙述逻辑，对已经确定的方案进行统一整理。方法包含两个核心模块：

- Omission-Aware Superpoint Bridging（OASB）：遗漏感知的超点语义桥接。
- Counterfactual Selective Recovery（CSR）：反事实选择性前景恢复。

二者在同一个元训练任务中联合执行，使用相同的区域语义表示，并通过统一损失优化。元测试执行完全一致的前向流程，仅不计算监督损失、不更新模型参数。

## 一、研究问题与 Motivation

这部分主要用于帮助学生理解研究出发点。正式论文中建议将详细 Motivation 放在 Introduction，Method 的 3.1 只保留必要概述。

### 1.1 问题背景

在小样本三维点云语义分割中，模型需要利用极少量已标注支持点云，识别未见类别在查询场景中的对应点。

由于支持样本数量有限，其提供的类别知识往往不能充分覆盖目标类别的几何形状、局部结构、点密度及遮挡变化。例如，支持集中可能只有完整、结构规则的椅子，而查询场景中的椅子可能具有不同的几何形状，甚至只呈现部分可见结构。

额外的无标签点云能够为类别知识扩展提供更多样本，但已有伪标签方法通常倾向于利用高置信度前景。这类区域更容易被模型识别，却不一定提供当前支持集所缺少的信息。

与此同时，真正具有互补几何特征的目标区域，可能因与支持样本差异较大而具有较低的前景置信度，甚至被高置信度地预测为背景。

因此，我们关注一个容易被忽略的矛盾：

> High Confidence Does Not Guarantee High Information Value.

即：高前景置信度不等于高信息价值。

这里的“信息价值”特指一个无标签区域相对于当前少量支持知识的补充作用，而不是信息论中严格定义的熵或互信息。

### 1.2 核心观察：A、B、C 三类区域

考虑当前任务需要分割类别 $c$，定义：

| 区域 | 含义 | 示例 |
| --- | --- | --- |
| A：支持锚点 | 具有真实标签的少量类别知识 | 完整、规则的椅子 |
| B：候选桥梁 | 已被模型可靠识别，但知识贡献尚不确定的无标签区域 | 具有不同局部几何的椅子 |
| C：遗漏候选 | 初始模型未识别为类别 $c$ 的区域 | 遮挡、截断或异形椅子，也可能是真实背景 |

对于某些真正的遗漏前景 C，可能存在：

$$
\operatorname{Sim}(A,C)
<
\operatorname{Sim}(B,C),
$$

即支持知识无法充分解释 C，而某个可信桥梁 B 可以提供更加匹配的局部语义信息。

但需要注意两个问题：

第一，并非所有高置信 B 都有价值。如果 B 与 A 高度重复，其补充作用可能很小。

第二，较强的 B–C 关联不等于语义正确。某些背景或其他类别区域可能具有与 B 相似的几何结构。

因此，合理的桥接机制必须同时回答：

1. 如何利用可信 B 建立 A 难以直接覆盖的语义关联？
2. 如何确认这种关联具有类别选择性，而不是错误地把相似背景扩展成前景？

对应地，我们设计 OASB 和 CSR 两个联合学习模块。

### 1.3 方法核心思想

少量支持点云 S + 独立辅助无标签点云 U

冻结初始分割器，生成支持条件化类别预测

A

真实支持锚点

B

可信桥梁候选

C

全部遗漏候选

模块一：OASB — 遗漏感知的超点语义桥接

计算 A→C 直接关联与 A+B→C 桥接关联，获得预测变化

模块二：CSR — 反事实选择性前景恢复

利用支持负对照评价 B 的选择性贡献，恢复可信 C

支持锚定的类别知识扩展 → 查询点逐点分割

在元训练中接受遗漏恢复、桥梁选择性和最终分割监督

# 3. Method

## 3.1 Problem Formulation and Preliminary

### 3.1.1 Semi-Supervised Few-Shot 3D Segmentation

考虑一个 $N$-way $K$-shot 半监督小样本三维语义分割任务：

$$
\mathcal T=(\mathcal S,\mathcal U,\mathcal Q).
$$

其中：

- $\mathcal S=\{(X_{c,k}^s,Y_{c,k}^s)\}$：支持集，每个目标类别具有 $K$ 个带标注支持样本。
- $\mathcal U=\{X_j^u\}_{j=1}^{M}$：独立辅助无标签点云集合。
- $\mathcal Q=\{(X^q,Y^q)\}$：训练时用于监督的独立查询集。
- $N$：当前任务的前景类别数。
- $M$：辅助无标签点云数。

一个点云表示为：

$$
X=\{(\mathbf p_i,\mathbf a_i)\}_{i=1}^{n},
$$

其中 $\mathbf p_i\in\mathbb R^3$ 表示点的空间坐标，$\mathbf a_i$ 表示颜色等可用属性，$n$ 为点数。

目标是利用 $\mathcal S$ 和 $\mathcal U$ 中的类别知识，为查询点云的每个点预测标签：

$$
\hat Y^q=f_\Theta(X^q\mid\mathcal S,\mathcal U).
$$

背景类别记为 $0$，当前目标类别记为 $1,\ldots,N$。

训练类别与测试类别互不重叠。测试时支持集提供真实标签，无标签集不提供真实标签，查询真值仅用于评估。

与标准小样本点云分割不同，本文显式引入独立辅助无标签点云。因此，实验时必须在相同协议下比较各方法，并避免支持、无标签和查询集合之间存在重复场景或重叠点云块。

### 3.1.2 Initial Support-Conditioned Predictor

首先，在基础类别上训练支持条件化点云分割器 $f_0$，获得三维编码器 $F_0$。

建议以 COSeg 为实现基础。预训练后冻结 $f_0$ 与 $F_0$。

对于辅助点云中的点 $i$，初始模型输出：

$$
P_0(c\mid i,\mathcal S),
$$

表示在当前支持集条件下，该点属于类别 $c$ 的初始概率。

初始模型主要承担两个作用：

- 区域语义可靠性估计：判断哪些无标签超点具备成为 B 的基本条件。
- 遗漏候选识别：找出尚未被识别为当前类别的 C。

需要强调，初始模型并不直接决定所有类别知识是否值得采用。其置信度只提供初始语义判断，而非最终信息价值。

### 3.1.3 Learnable Superpoint Representation

三维点云具有不规则空间分布，因此采用 Superpoint（超点） 作为无标签语义知识挖掘的基本单位。

首先，依据几何邻接和可用点属性，对点云进行类别无关的过分割：

$$
\mathcal R(X)=\{R_1,\ldots,R_L\},
$$

其中每个 $R_l$ 是包含若干局部相邻点的超点。

为了避免遗漏前景在区域构建阶段被排除，超点构建不得仅依据初始预测前景进行，而应覆盖完整的辅助点云。

对于点 $i$，初始编码特征为：

$$
\mathbf h_i=F_0(X)_i.
$$

引入可学习投影 $W_\phi$，得到：

$$
\mathbf z_i=
\operatorname{Norm}(W_\phi\mathbf h_i).
$$

进一步通过区域聚合构建超点特征：

$$
\boxed{
\mathbf z_R=
\operatorname{Norm}
\left(
\frac1{|R|}
\sum_{i\in R}\mathbf z_i
\right)
}
\tag{1}
$$

其中 $W_\phi$ 是第二阶段主要训练的参数。相同投影用于支持点、无标签超点和查询点，以保证不同数据集合处于统一的语义空间。

对于类别 $c$，利用支持点真实标签构建原型：

$$
\boxed{
\mathbf p_c^s=
\operatorname{Norm}
\left(
\frac{
\sum_{(X,Y)\in\mathcal S}
\sum_i\mathbf1[Y_i=c]\mathbf z_i
}{
\sum_{(X,Y)\in\mathcal S}
\sum_i\mathbf1[Y_i=c]
}
\right)
}
\tag{2}
$$

背景也构建对应支持原型。

这一部分属于点云表示基础，不作为独立创新模块。

## 3.2 Omission-Aware Superpoint Bridging

模块一：遗漏感知的超点语义桥接（OASB）

### 3.2.1 Motivation

少量支持点云通常只能表达有限的类内几何模式。

如果简单采用支持原型与所有无标签超点直接匹配，模型容易优先识别与支持样本外观接近的前景，却遗漏具有不同局部结构的真实同类超点。

因此，本模块不直接将高置信无标签区域合并到原型，而是首先建立两个可比较的预测条件：

- Direct Association：只利用支持原型 A 预测 C。
- Bridged Association：在真实支持 A 的基础上额外引入候选桥梁 B，再预测 C。

通过比较二者的差异，显式刻画 B 为当前任务补充的语义证据。

### 3.2.2 Bridge and Recovery Candidate Construction

对辅助点云超点 $R$，计算初始模型的区域类别概率：

$$
q_c(R)=
\frac1{|R|}
\sum_{i\in R}
P_0(c\mid i,\mathcal S).
$$

定义其初始预测类别：

$$
\hat c_0(R)=\arg\max_c q_c(R).
$$

对于当前目标类别 $c$，构建候选桥梁集合：

$$
\boxed{
\mathcal B_c=
\left\{
B\in\mathcal R(\mathcal U):
\hat c_0(B)=c,\ q_c(B)\ge\tau_b
\right\}
}
\tag{3}
$$

其中 $\tau_b$ 是基础类别验证阶段确定的最低桥梁可信度。

候选 B 具有基本可信的类别归属，但其真正的知识价值尚未确定。

进一步定义待恢复集合：

$$
\boxed{
\mathcal C_c=
\{C\in\mathcal R(\mathcal U):\hat c_0(C)\ne c\}
}
\tag{4}
$$

这里尤其重要：

即使一个超点被初始模型以很高置信度预测为背景，它仍然属于相应前景类别的 C 候选集合。

因此，方法不是传统意义上只处理低最大置信度的不确定点，而是尝试恢复低目标前景置信度的遗漏区域。

在计算桥接关系时，候选 B 与候选 C 来自不同的初始类别集合，不会发生同一超点对自身进行桥接的情况。

### 3.2.3 Support-Anchored Bridged Association

对于待恢复超点 $C$，利用真实支持原型计算类别 $c$ 的直接关联：

$$
\boxed{
s_c^0(C)=
\frac{
(\mathbf p_c^s)^\top\mathbf z_C
}{\tau}
}
\tag{5}
$$

其中 $\tau>0$ 为温度参数。

如果仅使用支持原型，则通过各类别 $s_c^0(C)$ 得到原始区域分类结果。

现在考虑将候选桥梁 $B\in\mathcal B_c$ 作为额外的类别语义参考，定义：

$$
\boxed{
s_c^B(C)=
\log
\frac{
\exp(s_c^0(C))
+
q_c(B)\exp(\mathbf z_B^\top\mathbf z_C/\tau)
}{
1+q_c(B)
}
}
\tag{6}
$$

其中：

- $s_c^0(C)$：支持原型对 C 的直接类别证据。
- $\mathbf z_B^\top\mathbf z_C/\tau$：B 与 C 的学习特征关联。
- $q_c(B)$：B 的初始类别可靠性。
- $s_c^B(C)$：引入 B 后的类别得分。

式（6）采用归一化的指数证据聚合。它不会无条件提高前景类别得分，而是根据 B 是否比原支持原型更加匹配 C，产生不同的预测变化。

定义：

$$
\boxed{
\Delta_c(B,C)=s_c^B(C)-s_c^0(C)
}
\tag{7}
$$

$\Delta_c(B,C)$ 表示在同一任务和固定模型状态下，引入 B 对 C 的类别预测所产生的变化。

其数学性质包括：

- 当 B 与 A 对 C 提供相同的关联得分时，$\Delta_c(B,C)=0$，意味着 B 未提供额外的关联优势。
- 当 B 对 C 的关联强于 A–C 直接关联时，$\Delta_c(B,C)>0$。
- 当 B 与 C 的关联较弱时，$\Delta_c(B,C)<0$。

这使桥梁知识不再简单等同于高置信度。

需要说明，$\Delta_c(B,C)$ 是模型内部的受控预测变化，而不是已经证明的真实因果效应或信息论信息增益。

### 3.2.4 模块输出与训练作用

本模块为每个类别和候选超点提供：

$$
\left\{
s_c^0(C),\
s_c^B(C),\
\Delta_c(B,C)
\right\}.
$$

这些结果继续进入模块二。

模块一没有单独训练一个 B–C 相似度分类器，而是使式（6）中的特征关联通过后续的遗漏恢复损失和查询分割损失共同优化。

因此，学习目标不是简单地让所有同类超点尽可能接近，而是使区域特征能够为当前支持知识尚未覆盖的前景提供有效补充，同时保留类别判别能力。

## 3.3 Counterfactual Selective Recovery

模块二：反事实选择性前景恢复（CSR）

### 3.3.1 Motivation

尽管模块一能够确定候选桥梁 B 是否提高了 C 的类别关联，但这种改善未必意味着正确的语义恢复。

在三维点云中，不同类别可能共享相似的局部几何结构。例如，椅腿与桌腿、柜体侧面与墙面等区域可能具有相似的超点特征。

因此，仅根据 $\Delta_c(B,C)>0$ 进行恢复，容易将具有相似结构的背景或其他类别错误地标记为当前前景。

为此，我们提出负对照约束的选择性桥梁验证：一个有效 B 不仅应当提高潜在遗漏前景的类别关联，还应避免无差别地增强其他类别。

### 3.3.2 Negative-Controlled Bridge Evaluation

在当前任务的真实支持点云中，提取不属于类别 $c$ 的超点作为负对照集合：

$$
\mathcal N_c=
\{D\mid D\text{ 为真实支持负类超点}\}.
$$

对于支持超点跨越多个语义类别的情况，采用真实支持标签裁剪负类区域，避免把包含类别 $c$ 点的混合超点错误作为负对照。

在 $N$-way 任务中，负对照可以包括背景及其他前景类别；在 1-way 任务中，主要来自真实支持背景。

利用式（7），计算候选桥梁 B 对负对照 D 的预测变化：

$$
\Delta_c(B,D)=s_c^B(D)-s_c^0(D).
$$

考虑到少量具有较大错误增强的负样本可能导致严重误恢复，定义风险敏感的负对照校正项：

$$
\boxed{
\eta_c(B)=
\log\left[
\frac1{|\mathcal N_c|}
\sum_{D\in\mathcal N_c}
\exp\big([\Delta_c(B,D)]_+\big)
\right]
}
\tag{8}
$$

其中 $[x]_+=\max(0,x)$。

该设计具有两个作用：

- 对预测变化为负的负对照不施加惩罚。
- 对发生较大错误前景增强的负对照给予更高权重。

进一步定义：

$$
\boxed{
V_c(B,C)=
\Delta_c(B,C)-\eta_c(B)
}
\tag{9}
$$

其中 $V_c(B,C)$ 表示经过负对照校正后的选择性桥梁贡献。

相较于单纯的 B–C 特征相似度，该指标同时衡量了桥梁带来的新增类别证据及其错误扩展风险。

需要强调，$\mathcal N_c$ 始终来自当前任务的真实支持点云，因此训练和测试均可以构建，不需要使用无标签点云的真实标签。

不过，有限的支持负对照无法覆盖所有未知干扰类别，因此 $V_c(B,C)$ 是一种有监督基础的选择性贡献估计，而非语义正确性的绝对保证。

### 3.3.3 Differentiable Bridge Selection

一个待恢复超点 C 可能由多个不同的 B 提供补充知识。

与为 B 预测一个固定的全局价值不同，我们采用 C 条件化的桥梁选择，使每个 C 根据当前任务的语义关系自适应利用不同 B。

对类别 $c$，定义：

$$
\boxed{
\alpha_c(B\mid C)=
\frac{
[V_c(B,C)]_+
}{
\kappa+
\sum_{B'\in\mathcal B_c}
[V_c(B',C)]_+
}
}
\tag{10}
$$

其中 $\kappa>0$ 为支持锚点保留项。

对应的原始支持预测权重为：

$$
\alpha_c^0(C)=
\frac{\kappa}{
\kappa+
\sum_{B'\in\mathcal B_c}[V_c(B',C)]_+
}.
$$

由此满足：

$$
\alpha_c^0(C)+
\sum_{B\in\mathcal B_c}\alpha_c(B\mid C)=1.
$$

式（10）具有三个重要特点。

第一，抑制无效桥梁。 当 B 的选择性贡献非正时，其权重严格为零。

第二，保留真实支持知识。 即使存在多个有效 B，原始支持预测仍具有非零权重。

第三，进行任务自适应选择。 桥梁权重依赖当前 C 的特征与类别关系，不需要额外训练 MLP 来预测跨任务知识价值。

该操作除 ReLU 拐点外几乎处处可微，因此可在元训练阶段通过损失监督学习区域特征空间。

### 3.3.4 Selective Foreground Recovery

根据各桥梁的选择性权重，聚合最终区域类别证据：

$$
\boxed{
\bar s_c(C)=
s_c^0(C)+
\sum_{B\in\mathcal B_c}
\alpha_c(B\mid C)\Delta_c(B,C)
}
\tag{11}
$$

对所有类别形成预测：

$$
\bar P(c\mid C)=
\frac{\exp(\bar s_c(C))}
{\sum_{k=0}^{N}\exp(\bar s_k(C))}.
$$

实际计算时，对辅助点云的所有超点都构建完整的多类别得分；如果当前区域本身属于某类别的桥梁候选，则从该区域的桥梁集合中排除自身，以避免自匹配。

背景类别不使用前景桥梁扩展，保留支持条件化的直接得分。

现在，对于某个初始未被识别为类别 $c$ 的超点 C，我们判断它是否能够被可靠恢复。

首先定义有效选择性贡献：

$$
g_c(C)=
\sum_{B\in\mathcal B_c}
\alpha_c(B\mid C)[V_c(B,C)]_+.
$$

进一步得到恢复权重：

$$
\boxed{
w_c(C)=
\mathbf1[\hat c_0(C)\ne c]\,
\mathbf1[\arg\max_k\bar s_k(C)=c]\,
\bar P(c\mid C)
\frac{g_c(C)}{1+g_c(C)}
}
\tag{12}
$$

其中：

- 第一项表示 C 确实未被初始模型识别为类别 $c$。
- 第二项要求桥接后在多类别竞争中将 C 预测为类别 $c$。
- 第三项是恢复后的类别概率。
- 第四项反映正向选择性桥梁证据的强度。

因此，只有当 C 具有可信的类别竞争结果和正向桥接贡献时，才会获得正的恢复权重。

这里恢复的是初始分类错误的已有三维点，而不是补全物理空间中缺失的点。

此外，当前方法只进行一次恢复，不允许新恢复的 C 再次参与后续多轮桥接传播，以降低错误扩展风险。

### 3.3.5 计算流程示例

假设当前任务需要分割类别“椅子”。

候选桥梁验证示例

| 桥梁 B | 对遗漏 C 的增量 | 负对照风险 | 净贡献 |
| --- | --- | --- | --- |
| B₁：重复外观 | 0.08 | 0.03 | 0.05 |
| B₂：互补几何外观 | 0.60 | 0.10 | 0.50 |
| B₃：与桌腿混淆 | 0.65 | 0.55 | 0.10 |

仅用于帮助学生理解，并非实际实验结果。

虽然 B₃ 对 C 的原始预测增强最大，但它也可能错误增强大量桌腿等负类超点。

因此，B₂ 获得更高的选择性贡献，更适合参与遗漏 C 的恢复。

这一步体现了整个研究的核心：桥梁价值不由其自身的前景置信度或单纯相似度决定，而由其在当前任务中提供的选择性补充证据决定。

## 3.4 Support-Anchored 3D Knowledge Augmentation

支持锚定的三维类别知识扩展

前两个模块已经能够从辅助无标签点云中寻找有效 B，并恢复潜在遗漏的 C。

接下来需要解决：这些恢复知识如何真正改善独立查询点云的分割结果？

我们采用简单的支持锚定类别参考集合，不引入第三个独立网络。

### 3.4.1 Bridge Contribution Aggregation

由于不同 B 对不同 C 的贡献不同，不能仅凭 B 的初始置信度确定其最终权重。

根据其对可信恢复超点的贡献，定义：

$$
\boxed{
w_c(B)=
q_c(B)
\frac{
\sum_{C\in\mathcal C_c}
\alpha_c(B\mid C)w_c(C)
}{
\epsilon+\sum_{C\in\mathcal C_c}w_c(C)
}
}
\tag{13}
$$

其中：

- $w_c(B)$：桥梁 B 在最终类别参考中的贡献权重。
- $q_c(B)$：B 的初始前景可靠性。
- $\alpha_c(B\mid C)$：B 对 C 的选择性权重。
- $w_c(C)$：C 的恢复可靠性。
- $\epsilon$：数值稳定常数。

如果一个 B 没有参与任何可靠恢复，其最终权重接近零，即使它自身具有很高的初始前景置信度。

这就把我们的 Motivation 直接落实到了类别知识扩展阶段。

### 3.4.2 Support-Anchored Class Memory

对于类别 $c$，构建补充参考集合：

$$
\boxed{
\mathcal M_c=
\{(\mathbf z_B,w_c(B))\}
\cup
\{(\mathbf z_C,w_c(C))\}
}
\tag{14}
$$

这里 $\mathcal M_c$ 只表示来自辅助无标签点云的补充知识。

真实支持原型 $\mathbf p_c^s$ 始终单独保留，避免错误伪标签完全取代真实支持语义。

对于每个类别，可以按照贡献权重保留少量补充超点，控制冗余和计算开销。

需要注意：

B 和 C 的角色不同。 B 由初始模型提供基本语义可信度，而 C 是通过 B 的选择性桥接才获得恢复资格。二者的最终权重分别反映其桥接贡献和恢复可靠性。

### 3.4.3 Query Point-Level Segmentation

对于查询点云 $X^q$ 中的点 $i$，使用共享特征空间计算：

$$
\mathbf z_i^q=
\operatorname{Norm}
\left(W_\phi F_0(X^q)_i\right).
$$

真实支持原型对查询点的直接类别得分：

$$
s_c^s(i)=
\frac{(\mathbf p_c^s)^\top\mathbf z_i^q}{\tau}.
$$

引入补充类别参考后，定义最终查询点类别得分：

$$
\boxed{
\ell_c^q(i)=
\log
\frac{
e^{s_c^s(i)}
+
\displaystyle\sum_{(\mathbf z_R,w_R)\in\mathcal M_c}
w_R e^{\mathbf z_R^\top\mathbf z_i^q/\tau}
}{
1+\displaystyle\sum_{(\mathbf z_R,w_R)\in\mathcal M_c}w_R
}
}
\tag{15}
$$

其中：

- $\ell_c^q(i)$：查询点 $i$ 属于类别 $c$ 的最终得分。
- $\mathbf p_c^s$：真实支持原型。
- $\mathbf z_R$：有效 B 或恢复 C 的超点特征。
- $w_R$：对应的知识贡献权重。

最终：

$$
P^q(c\mid i)=
\operatorname{Softmax}_c(\ell^q(i)),
\qquad
\hat y_i^q=\arg\max_c P^q(c\mid i).
$$

背景类别只采用原始支持参考，不添加前景桥梁。

该设计具有两个重要性质：

一是保留原始支持预测。 如果当前无标签池没有任何可靠补充知识，则式（15）自动退化为普通支持原型匹配。

二是最终进行逐点分类。 超点只用于辅助知识挖掘，查询点云中的每个点仍根据自身三维特征独立获得类别预测，不直接把整个超点硬赋为某个语义类别。

因此，OASB 和 CSR 恢复的类别知识真正作用于最终查询分割，而不是仅改善无标签点云自身的伪标签。

## 3.5 Episodic Optimization

统一元学习优化

前述两个模块不是分别训练的独立系统，而是在每个元训练任务中按照 OASB → CSR → 类别知识扩展 → 查询分割 的顺序联合执行。

整个方法采用两阶段训练策略。

### 3.5.1 Stage I：Initial Segmentation Pretraining

首先，在基础类别数据上训练支持条件化三维分割器 $f_0$。

该阶段沿用标准小样本点云语义分割的 episodic training：

1. 采样 $N$-way $K$-shot 支持集。
2. 利用支持标注构建类别原型。
3. 对有标注查询点云进行分割。
4. 通过查询交叉熵更新初始模型参数。

训练完成后冻结编码器 $F_0$ 及初始预测器 $f_0$。

这一阶段的目标是建立基础三维几何语义表征，而不是训练 OASB 或 CSR。

### 3.5.2 Stage II：Omission-Conditioned Joint Learning

第二阶段训练本文提出的两个模块。

每个基础类别元训练任务划分为：

$$
\mathcal T_{\mathrm{train}}
=
(\mathcal S,\mathcal U^m,\mathcal U^u,\mathcal Q).
$$

其中：

| 数据 | 作用 | 标签使用规则 |
| --- | --- | --- |
| $\mathcal S$ | 构建支持原型和负对照 | 使用真实标签 |
| $\mathcal U^m$ | 模拟无标签恢复任务 | 前向时隐藏真实标签，计算训练损失时使用 |
| $\mathcal U^u$ | 额外真正无标签点云，可选 | 全过程不使用真实标签 |
| $\mathcal Q$ | 独立查询分割监督 | 真实标签仅用于最终损失 |

将 $\mathcal U^m$ 和 $\mathcal U^u$ 共同作为辅助无标签池输入两个模块。

需要强调，真实标签不得用于提前筛选 B、确定 C 是否为真实前景或计算测试阶段无法获得的桥梁权重。

它们只能在完整前向传播之后用于计算损失。

这保证了训练和测试具有一致的数据使用规则。

### 3.5.3 Omission-Aware Recovery Loss

在基础类别元训练中，可以根据隐藏的真实标注判断哪些超点被初始模型遗漏。

对于模拟无标签超点 $R$，定义真实类别分布：

$$
y_R(c)=
\frac1{|R|}
\sum_{i\in R}
\mathbf1[Y_i=c].
$$

其优势是即使一个超点包含多个类别，也能够保留相应的监督信息。

设真实占比最高的类别为 $c_R^*$。采用遗漏感知权重：

$$
\omega_R=
1+\beta\,
\mathbf1[
c_R^*\neq0
\land
\hat c_0(R)\neq c_R^*
].
$$

其中 $\beta\ge0$ 控制对初始遗漏前景的关注程度。

定义：

$$
\boxed{
\mathcal L_{\mathrm{omit}}
=
-\frac1{|\mathcal R^m|}
\sum_{R\in\mathcal R^m}
\omega_R
\sum_{c=0}^{N}
y_R(c)\log\bar P(c\mid R)
}
\tag{16}
$$

其中：

- $\mathcal R^m$：模拟无标签点云中的超点集合。
- $\bar P(c\mid R)$：经过 OASB 和 CSR 后的区域分类概率。
- $\omega_R$：对遗漏前景进行适当强化的监督权重。

这一损失同时监督已识别前景、遗漏前景和背景，不会仅通过无限提高前景得分来优化恢复效果。

它直接驱动模块一学习适合遗漏恢复的区域语义关联。

### 3.5.4 Bridge Selectivity Loss

仅使用区域分类损失，模型仍可能学习到能够提高前景得分、但缺乏区分性的桥梁关系。

因此，引入桥梁选择性监督。

在模拟恢复集的真实标签中，构建：

- $\mathcal P_c$：初始没有被识别为 $c$、但真实属于 $c$ 的遗漏超点。
- $\mathcal D_c$：真实不属于 $c$ 的困难负类超点。

定义超点 R 的最大桥梁贡献：

$$
v_c(R)=\max_{B\in\mathcal B_c}V_c(B,R).
$$

希望 $v_c(C)$ 对真实遗漏前景较高，而对困难负类较低。

采用：

$$
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{select}}
={}&
\mathbb E_{C\in\mathcal P_c}
\operatorname{softplus}(-v_c(C))\\
&+
\mathbb E_{D\in\mathcal D_c}
\operatorname{softplus}(v_c(D)).
\end{aligned}
}
\tag{17}
$$

其中：

$$
\operatorname{softplus}(x)=\log(1+e^x).
$$

该损失不要求事先知道哪一个 B 最好，而是通过恢复目标的真实类别监督，让模型学习如何区分有效语义桥接与错误类别关联。

这项损失主要作用于模块二的选择性贡献机制，并将梯度传回模块一的区域表示参数。

如果当前类别不存在可信 B，则跳过该类别的选择性损失，不人为构造桥梁。

### 3.5.5 Query Segmentation Loss

根据式（15）获得最终查询点级预测：

$$
P^q(c\mid i)=
\operatorname{Softmax}_c(\ell^q(i)).
$$

利用查询点真实标签计算：

$$
\boxed{
\mathcal L_{\mathrm{seg}}
=
-\frac1{|\Omega_q|}
\sum_{i\in\Omega_q}
\log P^q(Y_i^q\mid i)
}
\tag{18}
$$

其中 $\Omega_q$ 为查询中的有效标注点集合。

这项损失保证由辅助无标签点云获得的 B、C 语义知识，最终必须对查询点云分割有帮助，而不仅仅是提高辅助点云自身的前景预测。

### 3.5.6 Overall Objective

最终训练目标为：

$$
\boxed{
\mathcal L=
\mathcal L_{\mathrm{seg}}
+\lambda_o\mathcal L_{\mathrm{omit}}
+\lambda_s\mathcal L_{\mathrm{select}}
}
\tag{19}
$$

其中：

- $\mathcal L_{\mathrm{seg}}$：最终查询分割损失。
- $\mathcal L_{\mathrm{omit}}$：遗漏前景恢复损失。
- $\mathcal L_{\mathrm{select}}$：桥梁选择性损失。
- $\lambda_o,\lambda_s$：对应权重。

第二阶段冻结初始三维编码器 $F_0$，主要更新共享投影 $W_\phi$。

模块二不额外引入大型网络，其选择性贡献和桥梁权重通过可微公式计算，优化信号可以反向传播至共享区域表示。

式（12）的硬类别判定不会对未被选中的参考产生梯度，但式（16）和式（17）仍然直接监督完整的区域桥接得分。这也是实现时需要保留三项损失的原因。

## 3.6 Meta-testing

在新类别测试时，仅给定：

$$
\mathcal T_{\mathrm{test}}=(\mathcal S,\mathcal U,\mathcal Q).
$$

模型不使用无标签集或查询集的真实标注，也不进行参数更新。

完整执行：

支持原型构建 → 超点生成 → 候选 B/C → OASB → CSR → 补充类别参考 → 查询逐点分割。

测试和训练采用完全相同的前向公式，区别仅在于测试阶段不计算式（16）—（19）的监督目标。

# 四、给学生的代码实现指南

这一部分建议作为学生阅读材料的实现附录，不直接放入正式论文 Method。

## 4.1 先明确各模块的输入输出

下面以单个 episode 为单位，设支持点数为 $P_s$，辅助点数为 $P_u$，查询点数为 $P_q$，辅助超点数为 $R$，投影特征维度为 $d$。

| 变量 | 张量形状 | 含义 |
| --- | --- | --- |
| `support_feat` | `[Ps, d]` | 支持点投影特征 |
| `unlabeled_feat` | `[Pu, d]` | 辅助点投影特征 |
| `query_feat` | `[Pq, d]` | 查询点投影特征 |
| `support_proto` | `[N+1, d]` | 当前任务支持类别原型 |
| `region_feat` | `[R, d]` | 辅助超点特征 |
| `initial_prob` | `[R, N+1]` | 初始模型的区域类别概率 |
| `bridge_score` | `[B, R]` | 某类别候选桥梁对全部超点的贡献 |
| `bridge_weight` | `[B, R]` | 选择性桥梁权重 |
| `recovery_score` | `[R, N+1]` | 恢复后的超点类别得分 |
| `query_logits` | `[Pq, N+1]` | 最终查询逐点类别得分 |

其中 B 是该类别保留的候选桥梁数量。

不同类别可以具有不同数量的 B，不必强行填充为完全相同的数量。

## 4.2 建议按六个程序组件实现

1. Point Feature Extraction & Superpoint Pooling

提取冻结的三维点特征，执行可学习投影及超点特征聚合。
2. Episode Candidate Builder

根据支持标签计算 A，根据初始预测产生 B 和 C，不使用 U 的真实标签。
3. OASB

批量计算 `s0`、`sB` 和 `delta`，对应式（5）—（7）。
4. CSR

计算负对照风险、净贡献、桥梁权重和恢复类别得分，对应式（8）—（12）。
5. Class Memory & Query Decoder

生成 B/C 补充参考及最终查询点级得分，对应式（13）—（15）。
6. Episodic Trainer

负责两阶段训练、模拟无标签监督、查询分割监督及模型评估。

## 4.3 一个 episode 的伪代码

Python
def forward_episode(support, unlabeled, query, training=True): # Stage II: frozen 3D backbone and initial segmentor with torch.no_grad(): hs, hu, hq = encode_F0(support, unlabeled, query) p0_u = initial_segmentor(support, unlabeled)
 # Shared trainable projection zs = normalize(W_phi(hs)) zu = normalize(W_phi(hu)) zq = normalize(W_phi(hq))
 # True support prototypes and support negative controls prototypes = build_support_prototypes(zs, support.labels) negatives = build_support_negative_regions(zs, support.labels)
 # Label-independent superpoint segmentation region_ids = build_superpoints(unlabeled) region_feat = pool_and_normalize(zu, region_ids) region_prob = pool_probabilities(p0_u, region_ids)
 scores = {} contributions = {} bridge_weights = {}
 for c in foreground_classes: # Select B from reliable predicted foreground B = select_bridge_candidates(region_prob, c)
 # Direct class evidence for every superpoint s0 = cosine_score(prototypes[c], region_feat)
 if len(B) == 0: scores[c] = s0 continue

这是结构性伪代码，不是可以直接运行的完整 PyTorch 文件。

实际实现时，`selectivity_loss` 需要使用未经 ReLU 截断的 $V_c(B,R)$ 计算式（17）的最大选择性贡献，并正确区分 $\mathcal P_c$ 与 $\mathcal D_c$。此外，训练恢复损失只能读取模拟无标签部分的真值，不能读取真正无标签部分的标签。

## 4.4 学生编码时最关键的五个检查

B、C 的生成只依赖初始预测，绝不能使用模拟无标签真值进行预筛选。负对照仅来自真实支持点云，训练与测试都能获得。如果没有任何有效 B，最终预测必须退化为仅使用支持原型。训练与测试的前向过程一致，且桥接损失能够向 W_phi 传播非零梯度。查询点云只能在最后进行分割，不参与辅助无标签池的 B/C 发现。

0/5 项完成

还有两个实现细节值得提前明确。

数值稳定性： 式（6）和式（15）中的指数聚合应使用 `torch.logaddexp` 或 `torch.logsumexp` 实现，避免直接计算大指数导致溢出。

计算复杂度： 桥梁与全部超点关联可通过矩阵乘法批量实现。建议先限制每类参与计算的 B 数量和最终记忆超点数量，使用初始置信度进行候选数量截断，但不要把置信度排序当作最终桥梁选择。

# 五、建议学生按照什么顺序开展实验？

不要一开始就训练全部模块，否则难以判断问题来自骨干网络、超点划分还是桥接算法。

| 阶段 | 要实现的版本 | 要验证的问题 |
| --- | --- | --- |
| Step 1 | COSeg 基础分割器 + 独立 U 协议 | 标准支持与查询分割能否正常复现？ |
| Step 2 | 高置信超点直接扩展原型 | 额外 U 是否有利用价值？ |
| Step 3 | 加入 OASB，不使用负对照 | 直接语义桥接能否恢复更多遗漏超点？ |
| Step 4 | OASB + CSR | 选择性干预能否抑制错误前景恢复？ |
| Step 5 | 完整训练损失 | 遗漏恢复和选择性监督是否带来稳定增益？ |

核心指标建议同时报告：

- Query mIoU：最终新类别查询分割性能。
- Missed-Foreground Recall：初始遗漏的真实前景中，被正确恢复的比例。
- False Recovery Rate：真实背景或其他类别被错误恢复为目标前景的比例。
- Bridge Utility Analysis：相同桥梁数量下，置信度选择、随机选择与 CSR 选择对最终查询性能的影响。

特别建议绘制一张遗漏前景恢复率与错误恢复率的关系曲线，比较普通高置信扩展、无负对照桥接及完整 CSR。这张图会直接支撑论文 Motivation 和第二模块的必要性。

## 六、最终论文结构与创新对应关系

| 论文部分 | 核心内容 | 在论文中的定位 |
| --- | --- | --- |
| Introduction | 高置信度不等于高信息价值；少量支持知识无法覆盖几何变化 | 科学问题 |
| 3.1 Problem Formulation | 半监督小样本点云任务、支持原型、超点表示 | 任务与基础 |
| 3.2 OASB | 支持直接预测与桥接预测、类别知识增量 | 核心模块一 |
| 3.3 CSR | 负对照风险、选择性桥梁权重、遗漏超点恢复 | 核心模块二 |
| 3.4 Knowledge Augmentation | 补充类别参考与查询点级分割 | 两模块的统一输出 |
| 3.5 Episodic Optimization | 三项损失联合优化 | 训练目标 |
| 3.6 Meta-testing | 相同前向流程、无梯度更新 | 简短说明 |

### 两项核心贡献的正式表述

Contribution 1 — Omission-Aware Superpoint Bridging. We develop an omission-aware superpoint bridging mechanism that explicitly models the change in class evidence induced by introducing reliable unlabeled superpoints beyond scarce labeled supports, enabling complementary geometric-semantic associations for overlooked foreground regions.

Contribution 2 — Counterfactual Selective Recovery. We introduce a negative-controlled selective recovery mechanism that evaluates the task-specific contribution of candidate bridges through controlled prediction comparisons, suppresses indiscriminate foreground expansion, and incorporates reliably recovered superpoints into support-anchored class representations.

### 最终给学生的核心理解

整篇论文实际只做一件事：

从无标签点云中寻找真正能够补充当前支持知识的超点，并利用它们恢复原本无法识别的前景。

其中：

- OASB 解决“能否建立补充关联”：比较 A 直接预测 C 与加入 B 后预测 C 的变化。
- CSR 解决“这种关联是否值得相信”：利用已知负类超点校正错误增强，并选择性恢复 C。
- 类别知识扩展解决“恢复结果如何发挥作用”：将有效 B 与可信 C 作为补充参考，改善查询点的最终语义分割。

本版本可作为学生的 3D-OCSB Method V1 实现依据。上述公式均为待实验检验的算法设计，尚非已验证的性能结论。正式投稿时，可将正文压缩为约 2.5–3 页的 Method，将超点生成细节、伪代码、张量形状和实现检查项移至补充材料。

如果你愿意，我可以：

-
-
-

---
