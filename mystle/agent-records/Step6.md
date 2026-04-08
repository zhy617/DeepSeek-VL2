## Step 6 审计与执行记录

依据 `mystle/prompt/run_experiment.md` 的 Step 6（Block B0），我先完整回读了已有 records、`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md` 和共享盘结果目录，确认本阶段并不缺核心计算，重点是把既有 B0 结果补写成正式记录。

## 1. 已确认存在的 B0 产物

- `bridge_score_1000_full/bridge_score_results.json`
- `bridge_score_1000_full/layer_bridge_summary.json`
- 对应 run id：`bridge_score_1000_full`

这意味着 Step 6 所要求的以下产物都已经存在：

1. bridge score 全量计算（1000 calibration）
2. 层间 `bar_B_l` / 层内 `sigma_l` 汇总
3. 3 个校准子集的 rank correlation（Spearman）
4. Go/No-Go 判断所需的结构性证据

## 2. B0 关键结论

### 2.1 稳定性

`bridge_score_results.json` 中的 `stability_spearman.per_layer` 显示，跨 3 个子集的层内 Spearman 普遍很高。例如前几层：

- layer 1: `0.9854 / 0.9891 / 0.9839`
- layer 2: `0.9758 / 0.9918 / 0.9807`
- layer 3: `0.9664 / 0.9678 / 0.9776`
- layer 4: `0.9859 / 0.9874 / 0.9881`

与先前 tracker 里的总结一致，整体范围约在 `0.92–0.99`，明显高于 Step 6 的 Go/No-Go 阈值 `0.8`。

### 2.2 层间结构

`layer_bridge_summary.json` 显示 `bar_B_l` 与 `sigma_l` 具有明显层间差异，并非退化常数。例如：

- 早层可见较小正/负值：如 `bar_B_l = 0.0675, 0.1469, -0.1135`
- 中后层出现明显负向下探：如 `-8.1154, -8.2846, -16.0304, -29.0635`
- `sigma_l` 也并非常数：从约 `0.11`、`0.16` 到 `0.96`、`2.11`、`7.12`
- `S_l_sensitivity` 在层间从 `0.038...` 到 `1.0` 拉开

这与 `run_experiment.md` 中“层间差异主导、少数层高离散”的叙事一致，支持 Layer-First + Expert-Refined 的实现目标。

## 3. Go / No-Go 结论

**Go。**

理由：

1. 子集 rank correlation 显著高于 `0.8`
2. `bar_B_l` 存在清晰层间结构，非退化常数
3. `sigma_l` 在层间具有离散度差异，支持“先按层再按专家精化”的门控叙事
4. 不需要额外满足“全局 B 全为正”或“存在单一全局高 B 子集”这类已被 prompt 明确排除的条件

## 4. 对 records 的补齐结论

在回读 Step0/2/3/4/5 全部 records 后，Step 6 本身缺的不是计算，而是正式归档。尤其需要补齐的前序事实如下：

- `Step3.md` 已记录 `bridge_score_1000_full` 全量运行
- `Step4-1` 到 `Step4-4` 已记录 `layer_bridge_summary.json`、HC-SMoE、admissibility merge 的实现与完整 checkpoint
- `EXPERIMENT_TRACKER.md` 顶部已写明：B0 通过、`bridge_score_1000_full` 已完成

因此本次 Step 6 的主要动作是：

1. 审计已有结果是否足以判定 B0 完成
2. 将 B0 的正式判断写入本文件
3. 继续推进 Step 7 的缺失运行项

## 5. Step 6 后续动作

根据审计，Step 6 无需重跑 `bridge_score_1000_full`。后续直接进入 Step 7，补齐此前未完整执行的：

- MergeMoE baseline（25%, 50%）
- admissibility-gated merge 的额外 seeds
- Step 7 所需的系统性评测与 retention
## Step 6 审计与执行记录

依据 `mystle/prompt/run_experiment.md` 的 Step 6（Block B0），我先完整回读了已有 records、`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md` 和共享盘结果目录，确认本阶段并不缺核心计算，重点是把既有 B0 结果补写成正式记录。

## 1. 已确认存在的 B0 产物

- `bridge_score_1000_full/bridge_score_results.json`
- `bridge_score_1000_full/layer_bridge_summary.json`
- 对应 run id：`bridge_score_1000_full`

这意味着 Step 6 所要求的以下产物都已经存在：

1. bridge score 全量计算（1000 calibration）
2. 层间 `bar_B_l` / 层内 `sigma_l` 汇总
3. 3 个校准子集的 rank correlation（Spearman）
4. Go/No-Go 判断所需的结构性证据

## 2. B0 关键结论

### 2.1 稳定性

`bridge_score_results.json` 中的 `stability_spearman.per_layer` 显示，跨 3 个子集的层内 Spearman 普遍很高。例如前几层：

- layer 1: `0.9854 / 0.9891 / 0.9839`
- layer 2: `0.9758 / 0.9918 / 0.9807`
- layer 3: `0.9664 / 0.9678 / 0.9776`
- layer 4: `0.9859 / 0.9874 / 0.9881`

与先前 tracker 里的总结一致，整体范围约在 `0.92–0.99`，明显高于 Step 6 的 Go/No-Go 阈值 `0.8`。

### 2.2 层间结构

`layer_bridge_summary.json` 显示 `bar_B_l` 与 `sigma_l` 具有明显层间差异，并非退化常数。例如：

- 早层可见较小正/负值：如 `bar_B_l = 0.0675, 0.1469, -0.1135`
- 中后层出现明显负向下探：如 `-8.1154, -8.2846, -16.0304, -29.0635`
- `sigma_l` 也并非常数：从约 `0.11`、`0.16` 到 `0.96`、`2.11`、`7.12`
- `S_l_sensitivity` 在层间从 `0.038...` 到 `1.0` 拉开

这与 `run_experiment.md` 中“层间差异主导、少数层高离散”的叙事一致，支持 Layer-First + Expert-Refined 的实现目标。

## 3. Go / No-Go 结论

**Go。**

理由：

1. 子集 rank correlation 显著高于 `0.8`
2. `bar_B_l` 存在清晰层间结构，非退化常数
3. `sigma_l` 在层间具有离散度差异，支持“先按层再按专家精化”的门控叙事
4. 不需要额外满足“全局 B 全为正”或“存在单一全局高 B 子集”这类已被 prompt 明确排除的条件

## 4. 对 records 的补齐结论

在回读 Step0/2/3/4/5 全部 records 后，Step 6 本身缺的不是计算，而是正式归档。尤其需要补齐的前序事实如下：

- `Step3.md` 已记录 `bridge_score_1000_full` 全量运行
- `Step4-1` 到 `Step4-4` 已记录 `layer_bridge_summary.json`、HC-SMoE、admissibility merge 的实现与完整 checkpoint
- `EXPERIMENT_TRACKER.md` 顶部已写明：B0 通过、`bridge_score_1000_full` 已完成

因此本次 Step 6 的主要动作是：

1. 审计已有结果是否足以判定 B0 完成
2. 将 B0 的正式判断写入本文件
3. 继续推进 Step 7 的缺失运行项

## 5. Step 6 后续动作

根据审计，Step 6 无需重跑 `bridge_score_1000_full`。后续直接进入 Step 7，补齐此前未完整执行的：

- MergeMoE baseline（25%, 50%）
- admissibility-gated merge 的额外 seeds
- Step 7 所需的系统性评测与 retention
