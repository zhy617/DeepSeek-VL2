# Experiment Plan

**Problem**: 多模态 MoE expert merging 中 bridge experts 被误合并导致跨模态能力退化  
**Method Thesis**: 通过模态错配干预发现 bridge experts 的交互效应，约束 merge admissibility  
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

- **Claim tested**: Bridge experts 可被稳定识别，且 bridge score 分布有意义
- **Why this block exists**: 没有这个 sanity check，整条线没有定义基础
- **Dataset/split/task**: 1000 条多模态校准样本（从 LLaVA-Bench + InfoVQA train 中采样）
- **Compared systems**: N/A（分析性实验）
- **Metrics**:
  - Bridge score 分布可视化（是否有清晰的高/低分群）
  - 跨 3 个不同校准子集的 rank correlation（目标 >= 0.8）
  - 高 bridge score 专家在不同层的分布
- **Setup details**: DeepSeek-VL2，3 种干预条件，routed-token zeroing 近似
- **Success criterion**: 存在稳定的高 bridge score 专家子集，且 rank correlation >= 0.8
- **Failure interpretation**: 若 bridge score 完全不稳定或无差异，当前 idea 需重新审视
- **Table/figure target**: Figure 1: bridge score 分布图 + 层间分布 + 稳定性分析
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
  - (a) Bridge-score-gated protection
  - (b) Routing-frequency-gated protection
  - (c) Activation-similarity-gated protection
  - (d) Random protection
  - (e) No protection (all merge)
- **Metrics**: bridge-sensitive performance retention
- **Setup details**: 固定总压缩率 50%，固定保护数量 = top-20% bridge score experts
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
| M0        | Sanity: bridge score 可计算且稳定                  | B0             | rank correlation >= 0.8 否则停      | 4-6h | 定义不稳       |
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
- **Risk**: Bridge score 分布无差异
  - **Mitigation**: 增加干预强度或使用不同 mismatch 策略
- **Risk**: 优势仅在特定 benchmark 上
  - **Mitigation**: 报告多个 benchmark，诚实讨论适用范围

## Final Checklist

- Main paper tables 覆盖 (Table 1, 2, 3)
- Novelty isolated (Block B3)
- Simplicity defended (无额外训练组件)
- Frontier contribution justified (interventional analysis for bridge discovery)
- Nice-to-have 与 must-run 分离

