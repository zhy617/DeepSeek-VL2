
## 已完成的实验

| 配置 | 状态 | 输出 |
|------|------|------|
| Admissibility 50%（`compression_keep=0.5`） | **完成** | `apply_global_merge_partition` + `vl_gpt.save_pretrained` + `DeepseekVLV2Processor.save_pretrained` |
| Admissibility 25%（`compression_keep=0.25`） | **完成** | 同上；日志中仍有约束聚类退化后的 **`clustering_fallback: unconstrained_cosine`** 警告（与 Step4-3 一致） |

**Checkpoint 目录（共享盘）：**

- 50%：`/root/fsas/vlm/deepseek-vl2-bridge/results/baselines/admissibility_keep_0p50/merged_model/`（`model-00001-of-00004.safetensors` … 共 4 分片 + `config.json`、processor 等）
- 25%：`/root/fsas/vlm/deepseek-vl2-bridge/results/baselines/admissibility_keep_0p25/merged_model/`

同目录下 **`merge_plan.json` / `meta.json` 已随本次完整跑重新写入**（与 Step4-3 的超参一致：`tau_disp auto_p75`、`max_sigma_layer`、`use_bz_quantile`、`bz_quantile_for_cutoff 0.75`、`delta_affinity 2.0`、`seed 42`）。

**日志：**  
`/root/fsas/vlm/deepseek-vl2-bridge/logs/admissibility_keep_0p50_fullmerge.log`、`admissibility_keep_0p25_fullmerge.log`

## 关键数字

- 50%：`n_routed` **64 → 32**
- 25%：**64 → 16**（含一次 UserWarning 回退无约束余弦聚类）

## 问题与处理

1. **首次在 `--smoke-forward` 下失败**：`language.model` 空序列 forward 时出现 **`RuntimeError: expected scalar type BFloat16 but found Float`**，合并与 `save_pretrained` 尚未执行即退出。  
2. **处理**：去掉 `--smoke-forward` 后两次完整跑均成功。  
3. **代码**：在 `admissibility_merge.py` 里把 smoke 用的 `dummy` 改为与 **`embed_tokens.weight` 相同 dtype**，便于以后加回 `--smoke-forward` 时再试。

## 文档

已更新 **`mystle/refine-logs/EXPERIMENT_TRACKER.md`**：Step 4 说明与 R006/R007 行已标明 **`merged_model/`  checkpoint 已产出**。

## 下一步建议（按 `run_experiment.md` Step 5）

用上述 `merged_model` 路径在 **`evaluate.py` / lmms-eval** 上跑 R006/R007 的 retention；若需多 seed，可对同一超参改 `--seed` 再各保存一份 `run_id`。