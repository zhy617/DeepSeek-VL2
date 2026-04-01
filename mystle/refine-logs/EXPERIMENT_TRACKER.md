# Experiment Tracker

- **数据与结果根目录**: 共享盘 `~/fsas` — 校准数据在 `datasets/deepseek-vl2-bridge/`，实验产出在 `vlm/deepseek-vl2-bridge/`（详见 `mystle/prompt/run_experiment.md`）。
- **Step 2 校准数据（已完成）**: `~/fsas/datasets/deepseek-vl2-bridge/calibration/` — 1000 条 InfoVQA 子样本（`Ahren09/info_vqa` train，seed=42），含 `original` / `visual_ablated` / `mismatch` 三种图像；清单 `manifest.jsonl`，元数据 `meta.json`。生成脚本：`mystle/experiments/prepare_calibration.py`。
- **Step 3 Bridge Score（已实现）**: `mystle/experiments/bridge_score.py` — MoE hook 计算 I_visual / I_text / I_mismatch、B、M，3 子集稳定性 Spearman，JSON + 图写入 `~/fsas/vlm/deepseek-vl2-bridge/results/<run_id>/`。多模态序列长不一致时默认 `--seq-align suffix`。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Bridge score computation | Bridge Score Module | 1k calibration | rank correlation, distribution | MUST | **脚本就绪** | 运行 `bridge_score.py` 填 Metrics |
| R002 | M1 | HC-SMoE baseline (25%) | HC-SMoE | InfoVQA/OCRBench/MMMU/MMBench | accuracy retention | MUST | TODO | |
| R003 | M1 | HC-SMoE baseline (50%) | HC-SMoE | same | accuracy retention | MUST | TODO | |
| R004 | M1 | MergeMoE baseline (25%) | MergeMoE | same | accuracy retention | MUST | TODO | |
| R005 | M1 | MergeMoE baseline (50%) | MergeMoE | same | accuracy retention | MUST | TODO | |
| R006 | M2 | Admissibility-gated merge (25%) | Ours | same | accuracy retention | MUST | TODO | 3 seeds |
| R007 | M2 | Admissibility-gated merge (50%) | Ours | same | accuracy retention | MUST | TODO | 3 seeds |
| R008 | M3 | Novelty: bridge-score gate | Bridge-score protection | InfoVQA/OCRBench | retention | MUST | TODO | fixed protection count |
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
