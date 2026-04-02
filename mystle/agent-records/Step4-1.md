
## 已完成的实验列表（状态）

| 项 | 状态 |
|----|------|
| **Step 3.9** `layer_bridge_summary.json` | 已完成（`bridge_score.py` 每次运行与 `bridge_score_results.json` 同目录写出） |
| **Step 4** HC-SMoE + Admissibility 合并 | 已完成（`moe_merge_core.py`、`baselines/hcsmoe_merge.py`、`admissibility_merge.py`） |
| **EXPERIMENT_TRACKER.md** | 已更新说明 Step 3.9 / Step 4 与用法 |

---

### Step 3.9：`layer_bridge_summary.json`

在 `bridge_score.py` 中增加 `compute_layer_bridge_summary()`，写出字段包括：

- 每层：`bar_B_l`、`sigma_l`、`B_z(e,l)`、`sigma_l_ge_tau_disp`（对比 `--tau-disp`）
- `S_l_sensitivity`：`|\bar{B}_l|` 在 MoE 层间的分位数（Layer-First 预算用）
- 原始 `B`、`M` 便于 Step 4 使用  

新增参数：`--tau-disp`（默认 `0.0`）、`--layer-summary-eps`（默认 `1e-8`）。

---

### Step 4：合并实现要点

1. **`moe_merge_core.py`**  
   - 跨 MoE 层对 expert 权重向量取平均 → 余弦距离 → 层次聚类（`average`/`complete`/`single`）。  
   - `resolve_target_n_routed`：满足 `n_group` 整除与 `num_experts_per_tok`。  
   - 对每层同一划分做 MLP 与 Gate（及 `e_score_correction_bias`）加权平均，并更新共享 `language.config.n_routed_experts`。

2. **`baselines/hcsmoe_merge.py`**（HC-SMoE baseline）  
   - `--compression-keep 0.25|0.5` 控制保留专家比例。  
   - `--no-save-model` 只写 `merge_plan.json`；否则 `save_pretrained` + Processor。

3. **`admissibility_merge.py`**（Ours）  
   - 必须提供 `--layer-bridge-summary` 指向 `layer_bridge_summary.json`。  
   - 在 `σ_l ≥ --tau-disp` 的层上检查 `B_z` 与 `|M_i−M_j|`（`--delta-affinity`）；默认用 `--use-bz-quantile` + `--bz-quantile-for-cutoff`，也可用绝对 `--tau-z`。  
   - 不可合并对在余弦距离矩阵上加大惩罚，再层次聚类并合并。

**示例命令：**

```bash
# 重新跑 bridge 以生成 layer_bridge_summary.json（若已有可略）
python mystle/experiments/bridge_score.py --results-dir ~/fsas/vlm/deepseek-vl2-bridge/results

python mystle/experiments/baselines/hcsmoe_merge.py \
  --compression-keep 0.5 --smoke-forward --output-dir ~/fsas/vlm/deepseek-vl2-bridge/results

python mystle/experiments/admissibility_merge.py \
  --layer-bridge-summary ~/fsas/vlm/deepseek-vl2-bridge/results/bridge_score_1000_full/layer_bridge_summary.json \
  --compression-keep 0.5 --tau-disp 0.0 --use-bz-quantile --smoke-forward
```

（若尚无 `layer_bridge_summary.json`，需对对应 `run_id` 再跑一次 `bridge_score.py`。）

---

### 关键说明

- **未在本机跑完整合并**：需下载/缓存 `deepseek-vl2-small` 与足够显存；CLI、`moe_merge_core` 导入与 `resolve_target_n_routed(64,0.5,8,6)→32` 已验证。  
- **设计约束**：各 MoE 层共用同一 `n_routed_experts`，故采用**全局同一 expert 划分**；Layer-First 的 `S_l` 已写入 JSON，后续可在 `admissibility_merge` 里加层敏感度权重（当前以 `B_z`/`M` 约束为主）。  
- **argparse**：`hcsmoe_merge` 中含 `%` 的 help 已写成 `%%`，避免格式化报错。

### 下一步建议

1. 对 `bridge_score_1000_full` 补跑 `bridge_score.py`（或只跑 `compute_layer_bridge_summary` 的离线脚本）以生成 `layer_bridge_summary.json`。  
2. 在 GPU 上跑 `hcsmoe_merge` / `admissibility_merge` 各压缩率，并用 Step 5 的 `evaluate.py` 做精度（若尚未实现则先做短句 `generate` smoke test）。