## Step 7 审计与执行记录

本阶段对应 `mystle/prompt/run_experiment.md` 的 Step 7 / Block B1。开始执行前，我先对已有 baseline、checkpoint、评测脚本和 tracker 做了审计，结论如下。

## 1. 审计结果：哪些已经做过，哪些还没做

### 已经完成

- `HC-SMoE` baseline 已有完整 25% / 50% checkpoint
  - `baselines/hcsmoe_keep_0p25/merged_model`
  - `baselines/hcsmoe_keep_0p50/merged_model`
- `admissibility-gated merge` 主 seed（`seed=42`）已完成 25% / 50%
  - `baselines/admissibility_keep_0p25/merged_model`
  - `baselines/admissibility_keep_0p50/merged_model`
- Step 5 的统一评测入口已存在
  - `mystle/experiments/evaluate.py`
  - `mystle/experiments/evaluate_mm.py`

### 开始 Step 7 时缺失

- `MergeMoE` baseline 25% / 50% 未实现，也没有结果
- `admissibility-gated merge` 还缺另外 2 个 seeds（要求总计 3 seeds）
- Step 7 的系统性评测结果与 retention 尚未补齐
- `EXPERIMENT_TRACKER.md` 中 R002/R003 仍写成 TODO，但实际 HC-SMoE checkpoint 已经存在，需要同步文档

## 2. 本次已完成的补齐

### 2.1 新增 MergeMoE baseline 实现

已新增：

- `mystle/experiments/baselines/mergemoe_merge.py`

实现方式：

- 与 HC-SMoE 共用现有 `moe_merge_core.py` 的合并与保存逻辑
- 聚类特征改为 **expert 输出空间签名**
- 对固定 probe hidden states 经过每个 expert MLP 的输出做跨层平均
- 再对这些输出签名做余弦距离 + 层次聚类

这对应 `EXPERIMENT_PLAN.md` 中的 “MergeMoE (output-space merge)” baseline。

### 2.2 环境修复

为了让 Step 7 能继续在 GPU 上执行，本次还修复了当前 `.venv` 与项目要求不一致的问题：

- `transformers` 从 `5.5.0` 调回仓库要求的 `4.38.2`
- `torch` 从 `2.11.0+cu130` 调回仓库要求的 `2.0.1+cu117`
- `numpy` 调回 `1.26.4`

额外做了两个最小兼容补丁，避免旧代码在中间过渡环境下导入失败：

- `deepseek_vl2/models/siglip_vit.py`
- `deepseek_vl2/models/modeling_deepseek.py`

修复后已验证：

- `torch 2.0.1+cu117`
- `transformers 4.38.2`
- `numpy 1.26.4`
- `torch.cuda.is_available() == True`

## 3. 本次已完成的新增运行

### 3.1 MergeMoE baseline

已完成：

- `baselines/mergemoe_keep_0p50`
  - `n_routed: 64 -> 32`
  - `seed=42`
  - `probe_count=32`, `probe_batch_size=8`
- `baselines/mergemoe_keep_0p25`
  - `n_routed: 64 -> 16`
  - `seed=42`
  - `probe_count=32`, `probe_batch_size=8`

说明：

- 两个 ratio 都已完整执行到 `merged_model/` 保存阶段
- `meta.json` 与 `merge_plan.json` 已写入各自目录

### 3.2 Admissibility 多 seed

已完成：

- `baselines/admissibility_keep_0p50_seed7`
  - `n_routed: 64 -> 32`
  - `seed=7`
  - 目录下已含 `merged_model/`, `meta.json`, `merge_plan.json`

进一步核对后发现：

- `baselines/admissibility_keep_0p50/merge_plan.json`
- `baselines/admissibility_keep_0p50_seed7/merge_plan.json`
- `baselines/admissibility_keep_0p50_seed123/merge_plan.json`

三者的 `groups` **完全一致**。也就是说，当前 `admissibility_merge.py` 的实现对不同 seed 实际上是**确定性的**，至少在 `compression_keep=0.5` 上如此；seed 变化不会改变最终 merge partition。

因此，Step 7 中“3 seeds”在当前实现下更接近**重复确认 deterministic result**，而不是产生 3 组不同 checkpoint。

### 3.3 正在继续补跑

截至本条记录写入时，仍需继续补齐：

- `admissibility_keep_0p25_seed7`
- `admissibility_keep_0p25_seed123`
- Step 7 的统一评测与 retention

补充说明：

- `admissibility_keep_0p50_seed123` 已成功写出 `meta.json + merge_plan.json`，且其 `merge_plan` 与 `seed=42/7` 完全一致；此前未生成完整 `merged_model/` 的原因是扩容前后的保存阶段异常，而不是 merge 结果不同。
- `MergeMoE` 的 `0.25 / 0.5` 两个 ratio 已在扩容后重新完整保存；此前共享盘满时遗留的半成品 checkpoint 已被替换。

## 4. 对 Step 7 当前状态的判断

若只从“merge checkpoint 是否存在”看：

- `HC-SMoE`：25% / 50% 已齐
- `MergeMoE`：25% / 50% 已齐
- `Ours`：25% / 50% 的主 seed 已齐，且额外 seed 正在补

但若严格按 `run_experiment.md` Step 7 的完整要求：

1. 运行 HC-SMoE baseline (25%, 50%)
2. 运行 MergeMoE baseline (25%, 50%)
3. 运行 admissibility-gated merge (25%, 50%, 3 seeds)
4. 评测所有配置
5. 更新 `EXPERIMENT_TRACKER.md`

则当前还 **不能算完全结束**，因为：

- Ours 的 3 seeds 还没全补齐
- 评测与 retention 还没完成
- tracker 尚未同步所有已完成项

## 5. 下一步执行顺序

基于截至当前的实际进度，Step 7 后续应按以下顺序继续：

1. 补跑 **基座多模态全量评测** `step5_base_mm_full`
   - 当前只有 `step5_mm_smoke`，不足以作为 Step 7 正式 `retention.json` 的 baseline
2. 串行补跑 `25%` 三组正式多模态评测
   - `HC-SMoE 25%`
   - `MergeMoE 25%`
   - `Ours 25%`
3. 对 `25% / 50%` 全部配置补齐相对基座的 `retention.json`
   - 若评测重新运行，则直接通过 `--baseline-json` 生成
   - 若不重跑，则可用 `evaluate.py retention` 离线补算
4. 基于当前 deterministic 行为，补齐 `admissibility_keep_0p25_seed7` 与 `admissibility_keep_0p25_seed123` 的 `meta/merge_plan`
5. 更新 `EXPERIMENT_TRACKER.md`
6. 回填本文件最终完成状态

## 6. 当前运行状态（追加）

截至当前：

- `50%` 三组正式多模态评测已经全部完成
  - `HC-SMoE 50%`
  - `MergeMoE 50%`
  - `Ours 50%`
- `25%` 三组正式多模态评测尚未开始，因此 Step 7 仍未结束
- `retention.json` 目前仍未补齐
  - 原因不是评测脚本不支持，而是**缺少基座多模态全量结果**作为正式 baseline
  - 目前共享结果中仅发现 `step5_mm_smoke/lmms_eval_results.json`，不足以作为正式 retention 基准
- 为补齐 retention，我已启动过基座多模态全量评测 `step5_base_mm_full`
  - 但在用户要求“先暂停测试，准备换卡继续”后，已手动停止该任务，等待后续在新卡上恢复
- 由于单卡限制，后续评测仍建议保持**串行**，避免多个大 checkpoint 同时加载造成显存与时间浪费

## 7. 已拿到的正式评测结果（追加）

### 7.1 HC-SMoE 50%

正式多模态评测已完成：

- run id: `step7_hcsmoe_0p50_mm`
- 结果文件：`mystle/tmp-results/step7_hcsmoe_0p50_mm/lmms_eval_results.json`

当前指标：

- `infovqa_val_lite` (`anls,none`): `0.4675836911477642`
- `mmmu_val` (`mmmu_acc,none`): `0.28222`
- `ocrbench` (`ocrbench_accuracy,none`): `0.631`

### 7.2 MergeMoE 50%

正式多模态评测已完成：

- run id: `step7_mergemoe_0p50_mm`
- 结果文件：`mystle/tmp-results/step7_mergemoe_0p50_mm/lmms_eval_results.json`

当前指标：

- `infovqa_val_lite` (`anls,none`): `0.2036050697100569`
- `mmmu_val` (`mmmu_acc,none`): `0.23889`
- `ocrbench` (`ocrbench_accuracy,none`): `0.126`

### 7.3 Ours 50%

正式多模态评测已完成：

- run id: `step7_admissibility_0p50_mm`
- 结果文件：`mystle/tmp-results/step7_admissibility_0p50_mm/lmms_eval_results.json`

当前指标：

- `infovqa_val_lite` (`anls,none`): `0.44764351589636936`
- `mmmu_val` (`mmmu_acc,none`): `0.31111`
- `ocrbench` (`ocrbench_accuracy,none`): `0.666`

### 7.4 当前 50% 结果的阶段性观察

在 `50%` 压缩率上，当前正式多模态结果显示：

- `Ours 50%` 在 `MMMU` 与 `OCRBench` 上优于 `HC-SMoE 50%`
- `HC-SMoE 50%` 在 `InfoVQA` 上略高于 `Ours 50%`
- `MergeMoE 50%` 三项指标均明显落后于前两者

但这里仍然只是**绝对分数**比较；由于正式基座多模态全量结果尚未补齐，当前还不能给出完整、规范的 retention 结论。

## 8. 暂停点（供换卡后续跑）

用户已明确要求：**先暂停测试，待更换 GPU 后继续**。

因此当前交接点如下：

- 已完成：
  - `50%` 三组正式多模态评测
  - `MergeMoE 25% / 50%` checkpoint
  - `HC-SMoE 25% / 50%` checkpoint
  - `Ours 25% / 50%` 主 seed checkpoint
  - `Ours 50%` 的 deterministic 多 seed 核对
- 已暂停：
  - `step5_base_mm_full`（基座多模态全量评测）
- 仍待继续：
  - `step5_base_mm_full`
  - `step7_hcsmoe_0p25_mm`
  - `step7_mergemoe_0p25_mm`
  - `step7_admissibility_0p25_mm`
  - `admissibility_keep_0p25_seed7` / `seed123` 的 `meta/merge_plan`
  - 所有配置的 `retention.json`
  - `EXPERIMENT_TRACKER.md` 最终同步
