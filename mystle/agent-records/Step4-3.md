
## 完成情况

已按 [Step4-2.md](mystle/agent-records/Step4-2.md) 的「下一步」在 GPU 上跑通 **`admissibility_merge.py`**（`--layer-bridge-summary` 指向全量 `layer_bridge_summary.json`），并参照 [run_experiment.md](mystle/prompt/run_experiment.md) 的 Step 4 要求补齐脚本能力与产物。

### 代码改动

1. **`mystle/experiments/admissibility_merge.py`**
   - 增加 **`--merge-plan-only`**：只写 `merge_plan.json` / `meta.json`，不写回权重（与 HC-SMoE 一致，避免 OOM）。
   - **`--tau-disp`** 改为字符串：`auto` / `auto_p50` / **`auto_p75`（默认）** / `auto_p90` 或显式浮点；`0.0` 会令几乎所有层同时施加 B_z/M，跨层 AND 极易导致聚类退化。
   - **`--admissibility-scope`**：`all` | **`max_sigma_layer`（默认）** —— 仅在 **σ_l 最大的一层** 上检查 B_z/M（原先对所有高 σ 层 AND，在实现上等价于几乎没有可合并对，`scipy` 在等距时会把 `fcluster` 收成 **1 簇**）。

2. **`mystle/experiments/moe_merge_core.py`**
   - 抽取 `_pair_ok_on_layer_row`；`pairwise_merge_admissible` 支持 `admissibility_scope`；`constrained_distance_matrix` 增加对应参数。
   - 若约束聚类仍退化为单簇，则 **回退** 为与 HC-SMoE 相同的无约束余弦聚类（并写入 `meta["clustering_fallback"]`）。

3. **环境**：当前 venv 曾缺 `scipy` 且 NumPy 2.x 与 torchvision 冲突，已安装 **`scipy`** 并将 **`numpy` 固定为 &lt;2**（与 `run_experiment.md` 一致）。

### 产出路径（`~/fsas`）

| 内容 | 路径 |
|------|------|
| 50% 压缩 admissibility | `~/fsas/vlm/deepseek-vl2-bridge/results/baselines/admissibility_keep_0p50/`（**约束聚类**，64→32，无 fallback） |
| 25% 压缩 admissibility | `~/fsas/vlm/deepseek-vl2-bridge/results/baselines/admissibility_keep_0p25/`（触发 **`clustering_fallback`**，64→16，与 HC-SMoE 余弦划分一致） |
| 两档合并汇总 | `~/fsas/vlm/deepseek-vl2-bridge/results/baselines/admissibility_all_compression.json` |

已更新 **`mystle/refine-logs/EXPERIMENT_TRACKER.md`**，并在 **`mystle/agent-records/Step4-2.md`** 末尾追加了上述执行记录。

### 后续建议

- **可推理合并权重**：去掉 `--merge-plan-only`，在显存足够时跑 **`apply_global_merge_partition` + `save_pretrained`**（或 CPU/offload）；当前 A100 80GB 可单独试跑，48GB 仍建议 plan-only。
- **评测（Step 5 / R006–R007）**：对 `merge_plan` 或合并后 checkpoint 跑 `evaluate.py` / lmms-eval。
- 若希望 **25% 也不走 fallback**：可再调 **`--bz-quantile-for-cutoff`**、`--tau-disp auto_p90`**，或略放宽 **`--delta-affinity`**，并在 `max_sigma_layer` 下重试。