# Experiment Tracker

- **数据与结果根目录**: 共享盘 `~/fsas` — 校准数据在 `datasets/deepseek-vl2-bridge/`，实验产出在 `vlm/deepseek-vl2-bridge/`（详见 `mystle/prompt/run_experiment.md`）。
- **Step 2 校准数据（已完成）**: `~/fsas/datasets/deepseek-vl2-bridge/calibration/` — 1000 条 InfoVQA 子样本（`Ahren09/info_vqa` train，seed=42），含 `original` / `visual_ablated` / `mismatch` 三种图像；清单 `manifest.jsonl`，元数据 `meta.json`。生成脚本：`mystle/experiments/prepare_calibration.py`。
- **Step 3 Bridge Score（已实现）**: `mystle/experiments/bridge_score.py` — MoE hook 计算 I_visual / I_text / I_mismatch、B、M，3 子集稳定性 Spearman，JSON + 图写入 `~/fsas/vlm/deepseek-vl2-bridge/results/<run_id>/`。**Step 3.9**：同目录写入 `layer_bridge_summary.json`（\(\bar{B}_l\)、\(\sigma_l\)、`B_z`、`S_l_sensitivity`、`sigma_l_ge_tau_disp` 等，供 admissibility merge）。多模态序列长不一致时默认 `--seq-align suffix`。
- **Step 4（已实现）**: `mystle/experiments/moe_merge_core.py`（合并算子）；`mystle/experiments/baselines/hcsmoe_merge.py`（HC-SMoE：权重余弦距离层次聚类 + 专家/Gate 加权平均，`--compression-keep 0.25|0.5`）；`mystle/experiments/baselines/mergemoe_merge.py`（MergeMoE：基于 fixed probe hidden states 的 **expert 输出空间签名** 聚类，`--probe-count` / `--probe-batch-size` 可配）；`mystle/experiments/admissibility_merge.py`（读取 `layer_bridge_summary.json`，在 **σ_l 最大的一层**上施加 \(B_z\)/\(M\) 约束，`--admissibility-scope max_sigma_layer` 默认；`--tau-disp auto_p75`；`--merge-plan-only` 与 HC-SMoE 一致省显存）。GPU 上已产出：`~/fsas/vlm/deepseek-vl2-bridge/results/baselines/hcsmoe_keep_0p25`、`hcsmoe_keep_0p50`、`mergemoe_keep_0p25`、`mergemoe_keep_0p50`、`admissibility_keep_0p25`、`admissibility_keep_0p50` 与相应汇总 JSON（其中 `admissibility_keep_0p25` 曾触发无约束回退，见对应 `meta.json` 的 `clustering_fallback`）。**2026-04-07**：环境已统一回项目要求（`torch 2.0.1`、`transformers 4.38.2`、`numpy 1.26.4`）并恢复 GPU 可用。
- **Step 5（基座 lm-eval，已完成）**: 仓库内 `lm-evaluation-harness` + `mystle/experiments/evaluate.py`、`mystle/experiments/deepseek_vl2_lm_eval.py`（注册 `deepseek_vl_v2`、模型类 `deepseek-vl2-hf`、修补 `gguf_file`/顶层 `use_cache` 等）。**基座** `deepseek-ai/deepseek-vl2-small` 在 **纯文本任务** `arc_easy` / `winogrande` / `piqa`（0-shot）上已跑满：**acc** 约 **77.9%** / **70.2%** / **77.5%**（详见 `~/fsas/vlm/deepseek-vl2-bridge/results/step5_base_lm_eval_full/lm_eval_results.json` 与 `meta.json`；日志 `logs/step5_base_lm_eval_full.log`）。
- **Step 5（基座 lmms-eval 多模态，已接入）**: 子模块 `lmms-eval/` + `mystle/experiments/evaluate_mm.py`、`mystle/experiments/deepseek_vl2_lmms.py`（运行时注册 `deepseek_vl2` → `DeepseekVL2LMMS`：`incremental_prefilling` + `generate`、`<image>` 与 PIL 对齐、与 lm-eval 相同的 `from_pretrained` 修补）。**依赖**：建议 `pip install -e ./lmms-eval`（见 `lmms-eval/pyproject.toml`，需 `torch>=2.1`、`transformers>=4.39` 等）；`PYTHONPATH` 需含仓库根与 `lmms-eval`。默认任务 `infovqa_val_lite,mmmu_val,ocrbench`；`mmbench_en_dev_lite` 可选（聚合阶段 GPT 判分需 `OPENAI_API_KEY` 等）。全量结果路径在运行后写入 `~/fsas/vlm/deepseek-vl2-bridge/results/<run_id>/`。
- **全量 1000 条（已完成）**: `run_id=bridge_score_1000_full` — 结果 `~/fsas/vlm/deepseek-vl2-bridge/results/bridge_score_1000_full/bridge_score_results.json`。实证：**层间 \(\bar{B}_l\) 差异大**；多数层 **\(\sigma_l\)** 小、少数层（如 L13/L19）\(\sigma_l\) 大；子集 Spearman 每层约 **0.92–0.99**（B0 通过）。**方法已改为 Layer-First + Expert-Refined**（见 `mystle/prompt/run_experiment.md`、`FINAL_PROPOSAL.md`）。重新跑 bridge 可生成 `layer_bridge_summary.json`；合并实验需 `--layer-bridge-summary` 指向该文件。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Bridge score computation | Bridge Score Module | 1k calibration | Spearman, \(\bar{B}_l\), \(\sigma_l\) | MUST | **完成（B0）** | `bridge_score_1000_full`；门控叙事已更新为 layer-first |
| R002 | M1 | HC-SMoE baseline (25%) | HC-SMoE | InfoVQA/OCRBench/MMMU/MMBench | accuracy retention | MUST | **merge + checkpoint 已产出** | `baselines/hcsmoe_keep_0p25`（`merged_model/`）；评测待跑 |
| R003 | M1 | HC-SMoE baseline (50%) | HC-SMoE | same | accuracy retention | MUST | **merge + checkpoint 已产出** | `baselines/hcsmoe_keep_0p50`（`merged_model/`）；评测待跑 |
| R004 | M1 | MergeMoE baseline (25%) | MergeMoE | same | accuracy retention | MUST | **merge + checkpoint 已产出** | `baselines/mergemoe_keep_0p25`（`merged_model/`；output-space signature baseline）；评测待跑 |
| R005 | M1 | MergeMoE baseline (50%) | MergeMoE | same | accuracy retention | MUST | **merge + checkpoint 已产出** | `baselines/mergemoe_keep_0p50`（`merged_model/`；output-space signature baseline）；评测待跑 |
| R006 | M2 | Admissibility-gated merge (25%) | Ours (layer-first + \(B_z\)@high-\(\sigma_l\)) | same | accuracy retention | MUST | **merge + checkpoint 已产出** | `baselines/admissibility_keep_0p25`（`merged_model/`）；评测与多 seed 待跑 |
| R007 | M2 | Admissibility-gated merge (50%) | Ours (同上) | same | accuracy retention | MUST | **merge + checkpoint 已产出** | `baselines/admissibility_keep_0p50`（`merged_model/`；约束聚类未退化）；评测与多 seed 待跑 |
| R008 | M3 | Novelty: bridge-score gate | 按层 \(B_z\) protection | InfoVQA/OCRBench | retention | MUST | TODO | 与主方法保护预算对齐 |
| R009 | M3 | Novelty: routing-freq gate | Routing-frequency protection | same | retention | MUST | TODO | |
| R010 | M3 | Novelty: activation-sim gate | Activation-similarity protection | same | retention | MUST | TODO | |
| R011 | M3 | Novelty: random gate | Random protection | same | retention | MUST | TODO | 3 seeds |
| R012 | M3 | Novelty: no protection | All merge | same | retention | MUST | TODO | |
| R013 | M4 | Naive merge + Router KD | HC-SMoE + Router KD | same | retention | MUST | TODO | |
| R014 | M4 | Ours without Router KD | Ours (no KD) | same | retention | MUST | TODO | |
| R015 | M4 | Ours + Router KD | Ours + KD | same | retention | MUST | TODO | |
| R016 | M5 | Compression sweep 12.5% | Ours | bridge-sensitive | retention | NICE | TODO | |
| R017 | M5 | Compression sweep 37.5% | Ours | bridge-sensitive | retention | NICE | TODO | |
| R018 | M5 | Failure analysis | qualitative | 20-30 examples | categories | NICE | TODO | |
