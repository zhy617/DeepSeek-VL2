# Research Proposal: Causally-Grounded Bridge Expert Discovery for Role-Aware Multimodal MoE Compression

## Problem Anchor

- **Bottom-line problem**: 现有 MoE expert merging 方法假设所有专家在同一比较空间中可合并，直接迁移到多模态 MoE 时导致跨模态能力（OCR、InfoVQA、复杂推理）严重退化
- **Must-solve bottleneck**: 缺少对"哪些专家允许被合并"（merge admissibility）的判断，特别是缺少对跨模态桥接专家（bridge experts）的操作化定义和保护机制
- **Non-goals**: 不做新的 merge 算子优化；不做模型预训练/架构设计；不做纯 router 修复
- **Constraints**: 1x RTX 4090 48GB GPU（算力平台）；3-4 个月到投稿；目标 ICLR 2027 / ICML 2026；数据与结果放在共享盘 `~/fsas`（见当前 `mystle/prompt/run_experiment.md`）
- **Success condition**: 在相同压缩率下，bridge-sensitive benchmark 上显著优于 role-agnostic merge，且该优势不能被 Router KD 解释

## Technical Gap

当前 MoE expert merging 文献（HC-SMoE、MergeMoE、PuzzleMoE、DM-MoE）全部默认专家是同质可比的。然而在多模态 MoE 中：

1. **角色异质性**：VEQ 和 FastMMoE 已证明视觉/文本专家重要性不对称；MoME/Uni-MoE 架构本身就分离了模态专家
2. **桥接脆弱性**：某些专家充当跨模态信息转译者（bridge experts），合并它们导致不可逆的跨模态对齐断裂
3. **因果可辨识性**：纯 routing 统计或激活相似度不足以区分"共享专家"和"桥接专家"；需要因果干预才能识别真正的跨模态中介作用

朴素方案（更大 merge ratio 或更好 merge 算子）不够，因为它们不回答"谁可以被合并"。Router KD 只做事后修补，不能恢复被合并掉的桥接结构。

## Method Thesis

- **One-sentence thesis**: 通过因果模态干预发现桥接专家的跨模态中介角色，将其转化为 merge admissibility 约束，使多模态 MoE 在相同压缩率下保留更多跨模态能力
- **Why smallest adequate intervention**: 不发明新 merge 算子，只在现有 merge pipeline 前加一个 role discovery + admissibility gate
- **Why timely**: 多模态 MoE（DeepSeek-VL2、Qwen3-VL MoE、InternVL3.5）正在成为主流部署对象，压缩需求迫切

## Contribution Focus

- **Dominant contribution**: 因果驱动的 bridge expert 操作化定义 + merge admissibility 框架
- **Supporting contribution**: 实证证据——same-role merge + bridge preservation 在 bridge-sensitive 任务上的结构性优势
- **Explicit non-contributions**: 不提出新 merge 算子；不解决通用 MoE routing 问题

## Proposed Method

### Complexity Budget

- **Frozen/reused**: 目标 MoE-MLLM 本体（DeepSeek-VL2）；现有 merge 算子（HC-SMoE / 简单平均）
- **New components**: (1) 因果干预 role discovery module，(2) admissibility-gated merge policy
- **Intentionally excluded**: 复杂的 router 重训练；多阶段渐进合并；Super Expert 保护（可作为 ablation）

### System Overview

```
Input: Pre-trained MoE-MLLM + Calibration data (200-500 multimodal samples)

Step 1: Causal Role Discovery
  For each expert e_i, layer l:
    Run calibration data with 4 conditions:
      (a) Original input (baseline)
      (b) Image masked (visual channel ablated)
      (c) OCR/text-in-image removed (bridge channel ablated)
      (d) Text paraphrased (language robustness check)
    Compute causal effect profile:
      CE_visual(e_i) = |output(a) - output(b)| when e_i is active
      CE_bridge(e_i) = |output(a) - output(c)| when e_i is active
      CE_text(e_i) = |output(a) - output(d)| when e_i is active
    Classify:
      Bridge: CE_bridge >> CE_visual and CE_bridge >> CE_text
      Visual-dominant: CE_visual >> CE_text
      Language-dominant: CE_text >> CE_visual
      Shared: balanced profile

Step 2: Merge Admissibility Gate
  A(e_i, e_j) = 1  iff  role(e_i) == role(e_j) AND neither is Bridge
  A(e_i, e_j) = 0  otherwise (inadmissible)
  
  Bridge experts: frozen (no merge) or merged at reduced ratio (e.g., 2x more conservative)

Step 3: Constrained Merge
  Apply existing merge algorithm (HC-SMoE clustering or simple averaging)
  only on admissible expert pairs within each layer

Step 4: Optional Router Repair
  Lightweight router-only KD on 200 calibration samples (optional, for fair comparison)

Output: Compressed MoE-MLLM with preserved bridge structure
```

### Core Mechanism

- **Input**: Pre-trained MoE-MLLM 权重 + 校准数据
- **Output**: Expert role labels + admissibility matrix + compressed model
- **Key novelty**: 因果干预定义 bridge expert（不是 routing 统计，不是激活相似度）
- **Why causal > non-causal**: routing overlap 或激活相似度无法区分"共享但不关键"vs"桥接且关键"的专家；因果干预直接测量模态移除后的影响

### Training Plan

- **无训练**：role discovery 和 merge 都是 training-free
- **可选微量训练**：Router KD 仅 200 样本，< 10 分钟
- **数据需求**：200-500 条多模态校准样本（从 benchmark 中采样）

### Failure Modes and Diagnostics

1. **因果干预成本高** → Mitigation: 只在 calibration set 上运行一次，按层并行
2. **Bridge role 不稳定** → Diagnostic: 跨不同 calibration 子集测量 role 一致性
3. **收益来自"少 merge"而非 bridge 保护** → Diagnostic: 固定保护数量，比较 bridge vs random vs high-frequency protection
4. **收益等价于 Router KD** → Diagnostic: naive merge + Router KD vs our method

### Novelty and Elegance Argument

- vs MergeMoE/HC-SMoE: 它们优化"怎么合并"；我们回答"该不该合并"
- vs VEQ: VEQ 证明异质性用于量化；我们把异质性转化为 merge constraint
- vs FastMMoE: FastMMoE 做 token pruning + expert activation reduction；我们做 role-aware expert merge
- vs "What Gets Activated": 它只在文本 MoE 做因果分析且不用于 merge；我们在多模态中做因果发现并用于 merge admissibility
- 机制最小化：不发明新 merge 算子，只在现有 pipeline 前加一个 gate

## Claim-Driven Validation Sketch

### Claim 1 (Primary): Bridge-aware merge > role-agnostic merge at equal compression

- **Minimal experiment**: 在 DeepSeek-VL2 上固定 25%/50% expert reduction
- **Baselines**: (1) role-agnostic HC-SMoE, (2) MergeMoE, (3) naive merge + Router KD
- **Metrics**: bridge-sensitive benchmarks (InfoVQA, OCRBench) 性能保持率 + 综合 benchmark (MMMU/MMBench)
- **Expected evidence**: bridge-sensitive 任务保持率显著更高（>=2% gain），综合任务至少持平

### Claim 2 (Supporting): Bridge protection 的收益不等价于 Router KD

- **Minimal experiment**: 比较 naive merge + Router KD vs same-role merge + bridge preserve (不加 Router KD)
- **Metrics**: bridge-sensitive 性能 + routing consistency
- **Expected evidence**: our method 在 bridge-sensitive 指标上仍优于 Router KD alone

### Claim 3 (Novelty isolation): 因果发现的 bridge > 非因果启发式

- **Minimal experiment**: 比较 causal bridge detection vs routing-based detection vs random protection
- **Metrics**: bridge-sensitive 性能保持率
- **Expected evidence**: causal > routing-based > random

## Experiment Handoff Inputs

- Must-prove claims: C1 (primary), C2 (supporting), C3 (novelty isolation)
- Must-run ablations: bridge protect vs random protect vs high-frequency protect
- Critical datasets: InfoVQA, OCRBench, MMMU/MMBench dev subsets
- Highest-risk assumption: bridge experts 可被因果干预稳定识别

## Compute & Timeline Estimate

- **Role discovery**: ~4-8 GPU-hours (500 samples × 4 conditions × forward pass)
- **Merge + eval per configuration**: ~2-4 GPU-hours
- **Total estimated**: ~30-50 GPU-hours for full experiment suite
- **Timeline**: Week 1-2: role discovery + sanity; Week 3-4: main experiments; Week 5-6: ablations; Week 7-8: paper writing
