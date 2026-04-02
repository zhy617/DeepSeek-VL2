# Research Proposal: Interventional Bridge Discovery for Merge-Admissible Multimodal MoE Compression

## Problem Anchor

- **Bottom-line problem**: MoE expert merging 假设所有专家可比；在多模态 MoE 中，这导致跨模态桥接路径被破坏，bridge-sensitive 任务（OCR、InfoVQA、复杂推理）严重退化
- **Must-solve bottleneck**: 缺少对 bridge experts 的操作化定义和 merge admissibility 判断
- **Non-goals**: 新 merge 算子；模型预训练/架构设计；纯 router 修复；通用因果发现
- **Constraints**: 1x RTX 4090 48GB GPU（算力平台）；3-4 个月到投稿；目标 ICLR 2027；数据集与实验产出放在共享盘 `~/fsas`（数据集与 `vlm/` 结果目录，见 `mystle/prompt/run_experiment.md`）
- **Success condition**: 在相同压缩率下，bridge-sensitive benchmark 上 >=2% 优于 role-agnostic merge，且该优势不能被 Router KD 或 random protection 解释

## Technical Gap

现有 merge 方法（HC-SMoE、MergeMoE、PuzzleMoE）全部假设专家在同一比较空间中可合并。多模态 MoE 中：
1. 专家角色异质——视觉/语言/桥接/共享功能不同
2. 桥接路径脆弱——bridge experts 被合并后，跨模态对齐不可逆断裂
3. 路由统计不足以识别 bridge——需要干预实验才能区分"共享"与"桥接"

## Method Thesis

- **One-sentence thesis**: 通过模态错配干预（modality mismatch intervention）发现专家的跨模态交互效应，将高交互效应专家标记为 bridge，以此约束 merge admissibility
- **Why smallest adequate**: 不改 merge 算子，只在 merge 前加一个 admissibility gate
- **Why timely**: DeepSeek-VL2、Qwen3-VL MoE 等大规模多模态 MoE 正面临部署压缩需求

## Contribution Focus

- **Dominant contribution**: Bridge score 定义（基于跨模态交互效应）+ merge admissibility gate
- **Supporting contribution**: 实证——admissibility-gated merge 在 bridge-sensitive 任务上的结构性优势
- **Explicit non-contributions**: 不提出新 merge 算子；不做通用因果发现框架

## Proposed Method

### Complexity Budget

- **Frozen/reused**: 目标 MoE-MLLM 本体；现有 merge 算子（HC-SMoE / weighted average）
- **New components**: (1) Bridge Score 计算模块，(2) Admissibility Gate
- **Intentionally excluded**: 复杂 router 重训练；多阶段渐进合并；硬分类 taxonomy

### System Overview

```
Pre-trained MoE-MLLM + Calibration Data (1k-2k multimodal pairs)
       │
       ▼
[Step 1] Bridge Score Computation (per expert, per layer)
       │  - 3 interventions: (a) original, (b) visual channel zeroed, (c) modality mismatch
       │  - B(e,l) = I_mismatch − (I_visual + I_text)/2（可为负；用于相对排序与层内标准化）
       │  - 层统计：\bar{B}_l、σ_l = std_e B(e,l)；层内 z：B_z(e,l)
       ▼
[Step 2] Layer-First + Expert-Refined Admissibility Gate
       │  - 层策略：由 \bar{B}_l（或跨层分位）定每层 merge 预算 / 冻结强度
       │  - 若 σ_l < τ_disp：该层主要靠层预算 + 相似度聚类，不强调 per-expert B
       │  - 若 σ_l ≥ τ_disp：per-expert 门控，A(e_i,e_j) 用 B_z 与 M，|M_i−M_j|<δ
       │  - 「高 bridge」：同层 B_z 高于分位 τ_z 的 expert 优先冻结（非全局常数 τ）
       ▼
[Step 3] Constrained Merge
       │  - Apply HC-SMoE (or any merge algorithm) subject to layer budgets + 上述 admissible pairs
       ▼
[Step 4] Compressed MoE-MLLM with preserved bridge structure
```

### Core Mechanism: Bridge Score via Modality Mismatch Intervention

**定义**：Bridge expert 不是对某个模态敏感的专家，而是对跨模态对齐敏感的专家。

**操作化**：

对每个 expert `e_i` 在层 `l`，用校准数据计算：

```
I_visual(e_i) = E_x[||h_l(x) - h_l(x_visual_ablated)||] when e_i is in top-K
I_text(e_i)   = E_x[||h_l(x) - h_l(x_text_ablated)||]   when e_i is in top-K  
I_mismatch(e_i) = E_x[||h_l(x) - h_l(x_mismatch)||]     when e_i is in top-K
```

其中 `x_mismatch` 是将图像替换为语义不相关图像（保持文本不变），或将文本替换为不相关文本（保持图像不变）。

**Bridge Score**:
```
B(e_i) = I_mismatch(e_i) - (I_visual(e_i) + I_text(e_i)) / 2
```

高 B(e_i) 意味着该专家对模态错配特别敏感——它的功能依赖于模态间的正确对齐，而非单一模态的存在。这才是真正的跨模态中介（cross-modal mediator）。

**Modality Affinity**:
```
M(e_i) = (I_text(e_i) - I_visual(e_i)) / (I_text(e_i) + I_visual(e_i) + ε)
```

### Admissibility Rule（Layer-First + Expert-Refined）

**动机（实证）**：在 DeepSeek-VL2-small 上，1000 条校准显示 **层间 \(\bar{B}_l\) 差异大、多数层内 \(\sigma_l\) 小、少数层 \(\sigma_l\) 大**。门控若仅用全局 \(\tau\) 区分每个 expert，与数据不符；故采用两层决策。

```
Algorithm: Layer-First Admissibility

Input:  {B(e,l), M(e,l)}，超参 τ_disp, τ_z, δ_affinity，及层敏感度映射（由 \bar{B}_l 到每层 merge 预算）
Output: 每层可合并对与冻结集

1. 对每个 MoE 层 l：计算 \bar{B}_l、σ_l，及 B_z(e,l) = (B(e,l) − \bar{B}_l)/(σ_l + ε)。

2. 层策略：按 \bar{B}_l 的跨层分位（或其它可报告规则）设定该层最大可合并比例或最小冻结比例。

3. 若 σ_l < τ_disp：该层在预算内按输出相似度/聚类合并；不按全局 B 阈值筛 expert。

4. 若 σ_l ≥ τ_disp：对 (e_i, e_j) 同层，
     A(e_i,e_j)=1 当且仅当 B_z(e_i,l)<τ_z 且 B_z(e_j,l)<τ_z 且 |M(e_i)−M(e_j)|<δ_affinity。
   B_z(e,l) > τ_z 的 expert 归入高-bridge 冻结候选（τ_z 取同层分位，使总保护数与 baseline 可比）。

5. 总压缩率固定时，各层预算与 (4) 中冻结集联合满足目标 expert 数。
```

### Practical Implementation

- **Cost reduction**: 使用 routed-token zeroing 近似因果效应，而非完整 forward pass 差异。对于每个 expert，只需在其被 route 到的 token 上计算效应，复杂度 O(|calibration| × top-K) 而非 O(|calibration| × |experts|)
- **Layer granularity**: 每层独立计算 B；**合并约束以层预算为第一轴**，expert 级规则仅在高 \(\sigma_l\) 层启用
- **Threshold selection**: 在验证集或校准集上网格搜索 \(\tau_{\mathrm{disp}}\)、\(\tau_z\)、\(\delta_{\mathrm{affinity}}\) 及层敏感度映射；报告敏感性分析

### Failure Modes and Diagnostics

1. **Bridge score 不稳定** → 跨不同 calibration 子集测量 Spearman；若 < 0.8 则增加样本量
2. **全局 B 多为负或层内 \(\sigma_l\) 小** → **不**单独判失败；改用 \(\bar{B}_l\) 分层 + **\(B_z\)** 与 \(\tau_{\mathrm{disp}}\)；叙事强调层间结构而非「全局正 B」
3. **收益来自"少 merge"** → 控制合并总数，比较 layer-first bridge-gated vs random-gated vs frequency-gated（同保护 expert 数）
4. **阈值敏感** → 报告 \(\tau_z\)、\(\tau_{\mathrm{disp}}\) 与层分位映射在若干分位下的下游表现

### Novelty and Elegance Argument

- **核心创新**：把 bridge expert 从直觉概念提升为可量化的交互效应（interaction effect），基于模态错配干预而非简单模态移除
- **vs "What Gets Activated"**: 它用因果效应区分 domain/driver experts 在文本 MoE；我们用模态错配干预区分 bridge experts 在多模态 MoE，且用于 merge constraint
- **vs VEQ**: VEQ 证明异质性用于量化；我们把交互效应用于 merge admissibility
- **vs Router KD**: Router KD 做事后修复；我们做事前约束
- **最小化**：不发明新 merge 算子，不做硬分类，不加 router 重训练

## Claim-Driven Validation Sketch

### Claim 1 (Primary): Admissibility-gated merge > role-agnostic merge at equal compression

- **Experiment**: DeepSeek-VL2 上 25%/50% expert reduction
- **Baselines**: (1) HC-SMoE (role-agnostic), (2) MergeMoE, (3) naive merge + Router KD
- **Metrics**: InfoVQA, OCRBench (bridge-sensitive) + MMMU/MMBench (综合)
- **Success**: bridge-sensitive >=2% gain, 综合至少持平

### Claim 2 (Novelty isolation): Bridge score > non-interventional heuristics

- **Experiment**: 固定保护 expert 数量，比较 bridge-score-gated vs routing-frequency-gated vs activation-similarity-gated vs random-gated
- **Metrics**: bridge-sensitive 性能保持率
- **Success**: bridge-score-gated 稳定最优

### Claim 3 (Supporting): Advantage != Router KD

- **Experiment**: naive merge + Router KD vs our method (no Router KD)
- **Metrics**: bridge-sensitive benchmarks
- **Success**: our method 仍优于 Router KD alone on bridge-sensitive tasks

## Compute & Timeline Estimate

- **Bridge score computation**: ~8-12 GPU-hours (1k samples × 3 conditions × cached forward)
- **Merge + eval per config**: ~2-3 GPU-hours
- **Full experiment suite**: ~40-60 GPU-hours
- **Timeline**: W1-2 role discovery + sanity → W3-4 main experiments → W5-6 ablations → W7-8 writing
