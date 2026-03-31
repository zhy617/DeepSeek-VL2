# Research Proposal: Interventional Bridge Discovery for Merge-Admissible Multimodal MoE Compression

## Problem Anchor

- **Bottom-line problem**: MoE expert merging 假设所有专家可比；在多模态 MoE 中，这导致跨模态桥接路径被破坏，bridge-sensitive 任务（OCR、InfoVQA、复杂推理）严重退化
- **Must-solve bottleneck**: 缺少对 bridge experts 的操作化定义和 merge admissibility 判断
- **Non-goals**: 新 merge 算子；模型预训练/架构设计；纯 router 修复；通用因果发现
- **Constraints**: 1x RTX 4090 48GB GPU（算力平台）；3-4 个月到投稿；目标 ICLR 2027
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
       │  - Compute interaction effect = output shift under mismatch - sum of unimodal shifts
       │  - Bridge Score = normalized interaction effect
       ▼
[Step 2] Admissibility Gate
       │  - Continuous bridge score B(e_i) ∈ [0, 1]
       │  - Modality affinity M(e_i) ∈ [-1, 1] (negative=visual, positive=language)
       │  - Admissible pair: A(e_i, e_j) = 1 iff B(e_i) < τ AND B(e_j) < τ AND |M(e_i) - M(e_j)| < δ
       │  - High-bridge experts (B > τ): frozen or merged at reduced ratio
       ▼
[Step 3] Constrained Merge
       │  - Apply HC-SMoE (or any merge algorithm) only on admissible pairs per layer
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

### Admissibility Rule

```
Algorithm: Merge Admissibility Gate

Input: Expert set E, Bridge scores {B(e_i)}, Modality affinities {M(e_i)}, 
       thresholds τ_bridge, δ_affinity
Output: Admissibility matrix A ∈ {0,1}^{|E|×|E|}

For each pair (e_i, e_j) in same layer:
  If B(e_i) < τ_bridge AND B(e_j) < τ_bridge AND |M(e_i) - M(e_j)| < δ_affinity:
    A(e_i, e_j) = 1  (admissible)
  Else:
    A(e_i, e_j) = 0  (inadmissible)

High-bridge experts (B > τ_bridge): frozen or merged with 2x conservative ratio
```

### Practical Implementation

- **Cost reduction**: 使用 routed-token zeroing 近似因果效应，而非完整 forward pass 差异。对于每个 expert，只需在其被 route 到的 token 上计算效应，复杂度 O(|calibration| × top-K) 而非 O(|calibration| × |experts|)
- **Layer granularity**: 每层独立计算 bridge score；高层（接近输出）的 bridge score 权重更高
- **Threshold selection**: 通过小量 validation set 上的 grid search 选择 τ_bridge 和 δ_affinity

### Failure Modes and Diagnostics

1. **Bridge score 不稳定** → 跨不同 calibration 子集测量 rank correlation；若 < 0.8 则增加样本量
2. **B(e_i) 退化为 I_visual + I_text** → 检查 B 的分布：若无高 B 专家，说明模型可能无明确 bridge 结构
3. **收益来自"少 merge"** → 控制合并总数，比较 bridge-gated vs random-gated vs frequency-gated
4. **阈值敏感** → 报告 τ_bridge 在 [10th, 20th, 30th percentile] 下的稳定性

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
