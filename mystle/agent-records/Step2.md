
## Step 2 已完成

已新增并跑通 **`mystle/experiments/prepare_calibration.py`**，从本地缓存的 **`Ahren09/info_vqa`**（`train`）里用 **seed=42** 随机抽 **1000** 条，并为每条写出三种图像版本：

| 子目录 | 含义 |
|--------|------|
| `images/original/` | 原始图文中的图 |
| `images/visual_ablated/` | 同尺寸 **RGB 全黑** 图 |
| `images/mismatch/` | **下一条**样本的图（环形移位 `k=1`，与 `mismatch_source_*` 字段一致） |

### 输出位置（约 3.9GB）

- 根目录：`~/fsas/datasets/deepseek-vl2-bridge/calibration/`（本机为 `/root/fsas/...`）
- **`manifest.jsonl`**：1000 行，每行含 `question`、`answer`、三路 **`paths`**、`image_size`、`mismatch_source_sample_id` 等
- **`meta.json`**：数据集名、条数、seed、mismatch 移位等（已补充 `image_format: png`）

已在 **`mystle/refine-logs/EXPERIMENT_TRACKER.md`** 顶部增加 Step 2 完成说明与路径。

### 脚本用法（后续重跑或换格式）

```bash
source .venv/bin/activate
export HF_DATASETS_CACHE="$HOME/fsas/datasets/deepseek-vl2-bridge/hf_datasets_cache"
python mystle/experiments/prepare_calibration.py \
  --num-samples 1000 --seed 42 \
  --output-dir "$HOME/fsas/datasets/deepseek-vl2-bridge/calibration"
# 更快、更小可加：--image-format jpeg --jpeg-quality 92
```

说明：`manifest` 每行写入后会 **`flush()`**，长时间跑也能看到进度；PNG 对超长信息图较慢，若需省空间/时间可用 `--image-format jpeg`。

### 下一步

按 **`run_experiment.md`**，可进入 **Step 3：实现 `bridge_score.py`**（三种干预里若需 **text_ablated**，一般在该步对 question 做替换/占位，与当前磁盘上的三路图像校准一起使用）。

若你希望，我可以直接按 `manifest.jsonl` 接上 DeepSeek-VL2 的 conversation 格式，起草 `bridge_score.py` 的加载与 batch 推理骨架。