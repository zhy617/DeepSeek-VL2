
## 已完成：Step 3 `bridge_score.py`

已在 `mystle/experiments/bridge_score.py` 实现 **Bridge Score** 与 **Modality Affinity**，并与 Step 2 的 `manifest.jsonl` / 三路图像对齐。

### 实现要点

1. **模型**：`AutoModelForCausalLM` + bf16，`prepare_inputs_embeds` 后只跑 `language.model`（不跑 `lm_head`），省显存。  
2. **MoE**：对每层 `DeepseekV2MoE` 注册 `gate` forward hook（`topk_idx`）与 `MoE` forward hook（子层输出 **h**，含 shared experts）。  
3. **四种输入**：`original`、`visual_ablated`、`mismatch`（mismatch 图 **resize** 到当前样本 `image_size`）、`text_ablated`（默认用与 question **同字符长度** 的占位串 `·`，可用 `--no-match-question-chars` 关闭）。  
4. **聚合位置**：在 `attention_mask` 与 **两路** `images_seq_mask` 的交集上累加（图像 token 区）；仅在 **原始 forward** 的 top-K 里出现的 expert 上计入对应差分。  
5. **序列长度**：多模态下不同图像可能导致视觉 token 数不同。默认 **`--seq-align suffix`**：对 `moe_out` / `topk` / mask 取 **末尾 `min(S)`** 对齐；`strict` 则长度不一致直接跳过。  
6. **子集稳定性**：按样本下标均分为 3 子集，算各子集 **B**，并给出层内 **Spearman**（`stability_spearman`）。  
7. **输出**：`~/fsas/vlm/deepseek-vl2-bridge/results/<run_id>/bridge_score_results.json` + 各层 B 直方图 + 层×expert 热力图；OOM 时可自动减半 batch（同一段逻辑里 `cur_bs //= 2`）。

### 依赖

本机已用 `pip install scipy matplotlib seaborn` 跑通；若环境里没有，请先安装（与 `run_experiment.md` 一致）。

### 示例命令

```bash
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$HOME/fsas/models/huggingface}"
python mystle/experiments/bridge_score.py \
  --calibration-dir "$HOME/fsas/datasets/deepseek-vl2-bridge/calibration" \
  --results-dir "$HOME/fsas/vlm/deepseek-vl2-bridge/results" \
  --batch-size 1
```

调试可加 `--max-samples 2`、`--skip-plots`（无 matplotlib 时）。

### 验证

已对 **1 条** 校准样本做 smoke test（`bridge_smoke_test3`），第 1 层 `pooled_all` 的 **64 个 expert 均有有限 B**；结果与图在  
`/root/fsas/vlm/deepseek-vl2-bridge/results/bridge_smoke_test3/`。

### 文档

已按 pipeline 在 `mystle/refine-logs/EXPERIMENT_TRACKER.md` 增加 Step 3 说明，并将 R001 标为脚本就绪（全量 1000 条跑完后再填具体指标）。

---

**说明**：在 **suffix 对齐** 下，orig 与干预序列在末尾 `L` 个位置上比较，这是为适配 VL 视觉 token 数随图像变化；若需严格逐 token 对齐，需固定预处理分辨率或改校准管线，可再单独迭代。