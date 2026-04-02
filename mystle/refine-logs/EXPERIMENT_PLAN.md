# Experiment Plan

**Problem**: 多模态 MoE expert merging 中 bridge experts 被误合并导致跨模态能力退化  
**Method Thesis**: 通过模态错配干预量化跨模态交互效应（Bridge score），以 **层优先、高离散层专家细化** 的 admissibility 约束 merge（见 `mystle/prompt/run_experiment.md` 实证与 Layer-First 门控）。  
**Date**: 2026-03-26  
**Target Model**: DeepSeek-VL2-small (MoE-MLLM)  
**Model Status**: 已验证可在 RTX 4090 48GB 上以 bf16 加载和推理

## Claim Map


| Claim         | Why It Matters                                                       | Minimum Convincing Evidence                                                    | Linked Blocks |
| ------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------- |
| C1 Primary    | Admissibility-gated merge > role-agnostic merge at equal compression | bridge-sensitive benchmarks >=2% gain at same compression rate                 | B1, B2        |
| C2 Novelty    | Bridge score > non-interventional heuristics                         | bridge-score gating outperforms routing-based, activation-based, random gating | B3            |
| C3 Supporting | Advantage is not just Router KD                                      | our method (no KD) > naive merge + Router KD on bridge-sensitive tasks         | B4            |
| Anti-claim    | All gains from "merging fewer experts"                               | control total merged count; report protection ratio                            | B3, B4        |


## Paper Storyline

- **Main paper must prove**: C1 (main result), C2 (novelty isolation), C3 (router independence)
- **Appendix can support**: threshold sensitivity analysis, additional model generalization, qualitative examples
- **Experiments intentionally cut**: multi-model sweep (beyond 2 models), training-based approaches, progressive merge stages

## Experiment Blocks

### Block B0: Bridge Score Sanity Check

- **Claim tested**: Bridge score **跨校准子集稳定**，且 **层间** \(\bar{B}_l\) 分层可解释（支撑 layer-first 门控）；**非**要求「全局高 B 子集」或「全体 B>0」
- **Why this block exists**: 没有这个 sanity check，整条线没有定义基础
- **Dataset/split/task**: 1000 条多模态校准样本（从 LLaVA-Bench + InfoVQA train 中采样）
- **Compared systems**: N/A（分析性实验）
- **Metrics**:
  - 每层 \(\bar{B}_l\)、层内 \(\sigma_l\)、全局 B 直方图与 **层×expert** 热力图
  - 跨 3 个不同校准子集的 **Spearman**（目标 >= 0.8）
  - **高 \(\sigma_l\) 层**上 expert 级 spread（与「多数层 \(\sigma_l\) 小」对照）
- **Setup details**: DeepSeek-VL2，3 种干预条件，routed-token zeroing 近似；门控实现以 **层内 \(B_z\)** + \(\tau_{\mathrm{disp}}\) 为准（见 `run_experiment.md`）
- **Success criterion**: 子集 Spearman >= 0.8 **且** 层间 \(\bar{B}_l\) 非退化（有明显层间差异）；与 VL2-small 实证一致时可声明「层间主导、少数层高离散」
- **Failure interpretation**: 若 rank correlation < 0.8 或层间无结构，则增加校准量或检查干预；**不**因「全局 B 多为负」单独判失败
- **Table/figure target**: Figure 1: 层均值 \(\bar{B}_l\) 曲线 / 表 + 热力图 + 稳定性分析
- **Priority**: MUST-RUN
- **Estimated cost**: ~4-6 GPU-hours

### Block B1: Main Anchor Result

- **Claim tested**: C1 — admissibility-gated merge 在同压缩率下优于 role-agnostic merge
- **Why this block exists**: 核心 reviewer belief change
- **Dataset/split/task**:
  - Bridge-sensitive: InfoVQA dev (500 samples), OCRBench (1000 samples)
  - General: MMMU dev (900 samples), MMBench dev (1000 samples)
- **Compared systems**:
  - (a) Uncompressed baseline
  - (b) HC-SMoE (role-agnostic merge)
  - (c) MergeMoE (output-space merge)
  - (d) Ours: admissibility-gated merge
- **Metrics**: accuracy / score retention (%), compression ratio, FLOPs reduction
- **Setup details**: 
  - 两个压缩率: 25% expert reduction, 50% expert reduction
  - DeepSeek-VL2 基座
  - 3 seeds for stochastic merge
- **Success criterion**: >=2% gain on bridge-sensitive at 25%, >=3% at 50%; general at least neutral
- **Failure interpretation**: 若无 bridge-sensitive 优势，bridge 叙事不成立
- **Table/figure target**: Table 1: Main Results
- **Priority**: MUST-RUN
- **Estimated cost**: ~8-12 GPU-hours

### Block B2: Compression Rate Sweep

- **Claim tested**: C1 在不同压缩率下是否一致
- **Why this block exists**: 证明优势是结构性的，非某个 ratio 的偶然
- **Dataset/split/task**: 同 B1，但增加 12.5% 和 37.5% 两个中间点
- **Compared systems**: HC-SMoE vs Ours
- **Metrics**: bridge-sensitive performance retention vs compression ratio curve
- **Setup details**: 单一 seed（成本控制）
- **Success criterion**: 所有压缩率下 Ours 的 bridge-sensitive 保持率 Pareto 更优
- **Failure interpretation**: 若只在特定 ratio 有优势，需分析原因
- **Table/figure target**: Figure 2: Pareto curve
- **Priority**: NICE-TO-HAVE
- **Estimated cost**: ~4-6 GPU-hours

### Block B3: Novelty Isolation — Bridge Score vs Heuristics

- **Claim tested**: C2 — bridge score 优于非干预式启发式
- **Why this block exists**: 拆解"因果干预是否必要"
- **Dataset/split/task**: 同 B1 bridge-sensitive 子集
- **Compared systems** (固定保护 expert 数量):
  - (a) Bridge-score-gated protection（**按层** top-20% **\(B_z\)** 或与主方法一致的层内分位；全局绝对 B 阈值仅作附录）
  - (b) Routing-frequency-gated protection
  - (c) Activation-similarity-gated protection
  - (d) Random protection
  - (e) No protection (all merge)
- **Metrics**: bridge-sensitive performance retention
- **Setup details**: 固定总压缩率 50%，保护预算与主方法对齐（**按层分配**后与 (a) 的总保护 expert 数可比）
- **Success criterion**: bridge-score > routing > activation > random，且差距 >= 1%
- **Failure interpretation**: 若 routing-based 同等有效，需降低 "interventional" 的卖点
- **Table/figure target**: Table 2: Protection Strategy Comparison
- **Priority**: MUST-RUN
- **Estimated cost**: ~6-8 GPU-hours

### Block B4: Router KD Independence

- **Claim tested**: C3 — 优势不等价于 Router KD
- **Why this block exists**: 防止被 Router KD 论文吞掉
- **Dataset/split/task**: 同 B1
- **Compared systems**:
  - (a) Naive merge + Router KD
  - (b) Ours (no Router KD)
  - (c) Ours + Router KD (upper bound)
- **Metrics**: bridge-sensitive + general benchmarks
- **Setup details**: Router KD 用 200 校准样本，1 epoch
- **Success criterion**: Ours (no KD) >= naive merge + Router KD on bridge-sensitive
- **Failure interpretation**: 若 Router KD 完全追回差距，论文需重构
- **Table/figure target**: Table 3: Router Independence
- **Priority**: MUST-RUN
- **Estimated cost**: ~4-6 GPU-hours

### Block B5: Failure Analysis

- **Claim tested**: 定性证据——bridge protection 确实保护了跨模态推理链路
- **Why this block exists**: 给 reviewer 机制证据
- **Dataset/split/task**: 20-30 个 InfoVQA/OCRBench 样例
- **Compared systems**: base vs naive merge vs ours
- **Metrics**: qualitative error categories, routing heatmap shift
- **Setup details**: 手动分析
- **Success criterion**: 能看到 naive merge 更多出现跨模态对齐断裂
- **Failure interpretation**: 若分析无法解释收益来源，论文会更像黑箱
- **Table/figure target**: Figure 3: Case analysis
- **Priority**: NICE-TO-HAVE
- **Estimated cost**: ~2 GPU-hours + manual analysis

## Run Order and Milestones


| Milestone | Goal                                         | Runs           | Decision Gate                    | Cost | Risk       |
| --------- | -------------------------------------------- | -------------- | -------------------------------- | ---- | ---------- |
| M0        | Sanity: B0（Spearman + 层间 \(\bar{B}_l\) 结构）   | B0             | Spearman >= 0.8 且层间非退化，否则停   | 4-6h | 定义不稳       |
| M1        | Baseline: 建立 HC-SMoE/MergeMoE 基线             | B1 (baselines) | 基线可复现                            | 4-6h | 工程问题       |
| M2        | Main: 跑 admissibility-gated merge            | B1 (ours)      | bridge-sensitive gain >= 2% 否则审视 | 4-6h | idea 不成立   |
| M3        | Novelty: 拆解 bridge score vs heuristics       | B3             | bridge-score 最优 否则降级叙事           | 6-8h | novelty 被吞 |
| M4        | Independence: Router KD 对照                   | B4             | 不被 Router KD 追回                  | 4-6h | 被吞         |
| M5        | Polish: compression sweep + failure analysis | B2, B5         | 可选                               | 6-8h | 低          |


## Compute and Data Budget

- **Total estimated GPU-hours**: 40-60 hours
- **Data preparation**: 
  - 1000 条校准样本（从 LLaVA-Bench + InfoVQA train 采样）
  - 评测集使用公开 dev split
- **Human evaluation**: 无需
- **Biggest bottleneck**: Bridge score 计算的 GPU 显存管理（三种干预条件的 forward pass）

## 存储与路径（共享盘 fsas）

- **共享盘根目录**: `~/fsas`（关机不丢；与系统盘分离）
- **顶层目录**: `datasets/`（数据）、`models/`（`HF_HOME` 模型缓存）、`pip-cache/`（pip）、`vlm/`（本类实验产出）
- **本实验约定**（与 `mystle/prompt/run_experiment.md` 一致）:
  - 校准数据与 HF datasets 缓存: `~/fsas/datasets/deepseek-vl2-bridge/`（含 `calibration/`、`hf_datasets_cache/`）
  - 实验结果与日志: `~/fsas/vlm/deepseek-vl2-bridge/`（含 `results/`、`logs/`）
- **仓库内**: 仅保留 `mystle/experiments/` 源码与 `mystle/refine-logs/` 文档；大文件不提交 git

## Risks and Mitigations

- **Risk**: Bridge score 计算过程中 GPU OOM（三种条件 × 多层 hook 数据）
  - **Mitigation**: 减小 batch size、分层计算、及时清理 GPU 缓存；已确认基础推理可行
- **Risk**: Bridge score 层内差异小、难以 per-expert 分群
  - **Mitigation**: 采用 **layer-first** 门控 + **\(B_z\)** 仅在高 \(\sigma_l\) 层；论文叙事与 `run_experiment.md` 一致
- **Risk**: 全局 B 多为负
  - **Mitigation**: 不依赖全局 B>0；用层内排序、\(\bar{B}_l\) 分层与 \(\tau_{\mathrm{disp}}\)
- **Risk**: 优势仅在特定 benchmark 上
  - **Mitigation**: 报告多个 benchmark，诚实讨论适用范围

## Final Checklist

- Main paper tables 覆盖 (Table 1, 2, 3)
- Novelty isolated (Block B3)
- Simplicity defended (无额外训练组件)
- Frontier contribution justified (interventional analysis for bridge discovery)
- Nice-to-have 与 must-run 分离

