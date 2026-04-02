

## 已完成内容

### 1. 离线补算 `layer_bridge_summary.json`（不重跑 bridge）

在 `bridge_score.py` 中增加了：

- `--layer-summary-only`：只读已有 `bridge_score_results.json`，写出同目录下的 `layer_bridge_summary.json`
- `--bridge-results-json`：指定该 JSON 路径

已在你的全量结果上执行：

```bash
python mystle/experiments/bridge_score.py --layer-summary-only \
  --bridge-results-json /root/fsas/vlm/deepseek-vl2-bridge/results/bridge_score_1000_full/bridge_score_results.json \
  --tau-disp 0.0
```

**产出**：`/root/fsas/vlm/deepseek-vl2-bridge/results/bridge_score_1000_full/layer_bridge_summary.json`（约 152KB）

---

### 2. Baseline 多压缩率 + 合并到 fsas

- 原先带 `--smoke-forward` 的完整合并会在 `apply_global_merge_partition` 时 **显存 OOM**（约 48GB 仍不够）。
- 在 `hcsmoe_merge.py` 中增加了 **`--merge-plan-only`**：只做聚类并写入 `merge_plan.json` / `meta.json`，**不把合并写回整模**，显存可承受；适合你要的「各压缩率方案 + 汇总」。

已生成：

| 路径 | 说明 |
|------|------|
| `.../results/baselines/hcsmoe_keep_0p25/` | `compression_keep=0.25`，64→16 专家 |
| `.../results/baselines/hcsmoe_keep_0p50/` | `compression_keep=0.5`，64→32 专家 |
| `.../results/baselines/hcsmoe_all_compression.json` | 上述两次 run 的 **合并汇总**（含各 `meta` 与完整 `merge_plan`） |

合并脚本：`mystle/experiments/baselines/merge_baseline_runs.py`

```bash
python mystle/experiments/baselines/merge_baseline_runs.py -o \
  ~/fsas/vlm/deepseek-vl2-bridge/results/baselines/hcsmoe_all_compression.json \
  ~/fsas/vlm/deepseek-vl2-bridge/results/baselines/hcsmoe_keep_0p25 \
  ~/fsas/vlm/deepseek-vl2-bridge/results/baselines/hcsmoe_keep_0p50
```

---

### 3. `admissibility_merge` 用法提示

`--layer-bridge-summary` 可指向：

`/root/fsas/vlm/deepseek-vl2-bridge/results/bridge_score_1000_full/layer_bridge_summary.json`

---

### 问题与处理

- **OOM**：全量 `apply` + smoke 会爆显存 → 用 `--merge-plan-only` 只导出合并方案；若以后要 **可推理的合并权重**，需在更大显存或 CPU/offload 策略下再去掉该选项并保存模型。

---

### 下一步（如需）

在 GPU 上跑 **`admissibility_merge.py`**（带 `--layer-bridge-summary` 指向上面的 `layer_bridge_summary.json`），或对 HC-SMoE 做 **完整合并 + `save_pretrained`** 时再单独开任务并控显存。

---

### 4. 已执行：Admissibility merge（Step4 下一步）

- 脚本增加 **`--merge-plan-only`**；默认 **`--tau-disp auto_p75`**、**`--admissibility-scope max_sigma_layer`**（仅在 σ_l 最大的一层上做 B_z/M 门控，避免跨层 AND 导致距离矩阵全惩罚、聚类退化为单簇）。
- 已写入共享盘：
  - `~/fsas/vlm/deepseek-vl2-bridge/results/baselines/admissibility_keep_0p50/`（约束聚类 → 32 专家）
  - `~/fsas/vlm/deepseek-vl2-bridge/results/baselines/admissibility_keep_0p25/`（强压缩下触发 `clustering_fallback`，划分等效 HC-SMoE 余弦聚类 → 16 专家，见 `meta.json`）
  - 汇总：`~/fsas/vlm/deepseek-vl2-bridge/results/baselines/admissibility_all_compression.json`