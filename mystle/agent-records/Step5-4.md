
## 已完成：Step 5 测评脚本

依据 `run_experiment.md` Step 5、`Step5-3.md` 的数据集约定，以及 `Step5-1`/`Step5-2` 里对 lm-eval / lmms-eval 的衔接说明，已在 **`mystle/experiments/evaluate.py`** 做成**统一入口**，并保留 **`evaluate_mm.py`** 作为多模态兼容封装。

### 功能概览

| 能力 | 说明 |
|------|------|
| **`--backend text`** | lm-eval，默认任务 `arc_easy,winogrande,piqa`（与旧 CLI 兼容） |
| **`--backend mm`** | lmms-eval，默认任务 **`infovqa_val_lite,mmmu_val,ocrbench`**（与 Step5-3 / `run_experiment.md` 一致） |
| **位置参数** | `python evaluate.py text ...` / `python evaluate.py mm ...` 等价于带对应 `--backend` |
| **`device`** | 多模态路径已传入 **`lmms_eval.simple_evaluate(..., device=...)`**（修正此前 `evaluate_mm.py` 未传 device 的问题） |
| **`--batch-size`** | 未指定时 **text=4，mm=1**（与 VL2 lmms 适配器一致） |
| **Score retention (%)** | `--baseline-json` 指向基座 `lm_eval_results.json` 或 `lmms_eval_results.json`，评测结束后在当次 run 目录写 **`retention.json`** |
| **离线 retention** | `python evaluate.py retention --baseline ... --current ...` |
| **MMBench** | 仍通过 `--tasks` 追加（如 `mmbench_en_dev_lite`）；需 GPT 判分时自备 `OPENAI_API_KEY`（文档已说明） |

### 环境与路径（与 Step5-3 一致）

模块文档字符串中补充了 **`HF_DATASETS_CACHE` / `LMMS_EVAL_DATASETS_CACHE` / `HF_TOKEN` / `PYTHONPATH`** 的推荐写法。

### `evaluate_mm.py`

现为薄封装：在参数前插入 **`mm`**，等价于 `python evaluate.py mm ...`。

### 复现命令示例

```bash
source /root/DeepSeek-VL2/.venv/bin/activate
export HF_HOME=~/fsas/models/huggingface
export HF_DATASETS_CACHE=~/fsas/datasets/deepseek-vl2-bridge/hf_datasets_cache
export LMMS_EVAL_DATASETS_CACHE="$HF_DATASETS_CACHE"
export PYTHONPATH=/root/DeepSeek-VL2:/root/DeepSeek-VL2/lmms-eval

# 多模态全量（调试可加 --limit 5）
python /root/DeepSeek-VL2/mystle/experiments/evaluate.py mm --run-id step5_base_mm

# 合并模型相对基座算 retention
python .../evaluate.py mm --pretrained /path/to/merged --baseline-json ~/fsas/.../base/lmms_eval_results.json --run-id merged_eval
```

若你希望把 **`mystle/refine-logs/EXPERIMENT_TRACKER.md`** 里 Step 5 小节改成与上述 CLI 完全一致，可以说一下我可以直接改一版文档。