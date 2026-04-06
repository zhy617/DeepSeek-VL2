
已按 `Step5-2.md` 里默认评测任务 **`infovqa_val_lite,mmmu_val,ocrbench`**，用 `/root/hf_token.txt` 中的 token，通过 **`datasets.load_dataset`**（与 lmms-eval 的加载方式一致）完成预下载与缓存生成。

## 已下载的数据集

| 任务 | HuggingFace 路径 | 说明 |
|------|------------------|------|
| **infovqa_val_lite** | `lmms-lab/LMMs-Eval-Lite`，子集 **`infovqa_val`** | 含 **`lite`** split（约 500 条） |
| **mmmu_val** | **`lmms-lab/MMMU`** | 全量 `DatasetDict`：`dev` / **`validation`**（评测用）/ `test`；`load_dataset` 会拉整库，因此时间较长 |
| **ocrbench** | **`echo840/OCRBench`** | 含 **`test`** split |

## 缓存位置

与 `run_experiment.md` / `Step5-2` 建议一致：

- **`HF_DATASETS_CACHE`**：`~/fsas/datasets/deepseek-vl2-bridge/hf_datasets_cache`（当前机器上约 **19G**）
- **`HF_HOME`**：`~/fsas/models/huggingface`（已创建目录；本次主要占用在 datasets 缓存）

运行评测前请保持：

```bash
export HF_HOME=~/fsas/models/huggingface
export HF_DATASETS_CACHE=~/fsas/datasets/deepseek-vl2-bridge/hf_datasets_cache
export HF_TOKEN="$(tr -d ' \n\r' < /root/hf_token.txt)"
export PYTHONPATH=/root/DeepSeek-VL2:/root/DeepSeek-VL2/lmms-eval
```

（`lmms-eval` 也会读 `LMMS_EVAL_DATASETS_CACHE`；与 `HF_DATASETS_CACHE` 设为同一目录即可。）

**说明**：`mmmu_val` 只需 **`validation`**，但 Hub 上 `load_dataset("lmms-lab/MMMU")` 会构建/缓存所有 split（含大 **`test`**），这是正常现象，之后离线评测只要缓存完整即可。