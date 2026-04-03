下面是已在仓库里接好的 **lmms-eval 多模态 Step 5** 说明与用法。

## 新增文件

1. **`mystle/experiments/deepseek_vl2_lmms.py`**  
   - 实现 lmms-eval 的 **`DeepseekVL2LMMS`**（`@register_model("deepseek_vl2")`）。  
   - 与 `deepseek_vl2_lm_eval.py` 一致：注册 `deepseek_vl_v2`、去掉 `from_pretrained` 里为 `None` 的 `gguf_file` / `quantization_config`、补顶层 `use_cache`。  
   - **`generate_until`**：`doc_to_visual` → PIL；若无 `<image>` 则按图像数量前缀 `<image>`；`DeepseekVLV2Processor` → **`incremental_prefilling(chunk_size=512)`** + **`generate`**；支持任务里的 `until`、`max_new_tokens` 等。  
   - `loglikelihood` / `generate_until_multi_round`：未实现（与部分内置模型一致，当前默认任务为 `generate_until`）。

2. **`mystle/experiments/evaluate_mm.py`**  
   - 把 `lmms-eval` 子目录加入 `sys.path`，并向 **`MODEL_REGISTRY_V2`** 注册 **`deepseek_vl2`** → 上述类。  
   - 调用 **`lmms_eval.evaluator.simple_evaluate`**。  
   - 默认任务：**`infovqa_val_lite,mmmu_val,ocrbench`**（与 `run_experiment.md` 对齐）。  
   - **`--save-full`** 才写入 `configs`（否则 JSON 更小）。  
   - **注意**：不要在 `model_args` 里再写 `device` / `batch_size`，避免与 `simple_evaluate` 传入的附加参数重复（已修正）。

## 依赖与环境

- 建议在本仓库根目录执行：**`pip install -e ./lmms-eval`**，并按 **`lmms-eval/pyproject.toml`** 满足 **`torch>=2.1`、`transformers>=4.39`** 等（当前 `.venv` 若仍是 torch 2.0 / transformers 4.38，会与声明冲突，需升级后再跑全量）。  
- 运行前：**`export PYTHONPATH=/root/DeepSeek-VL2:/root/DeepSeek-VL2/lmms-eval`**。  
- 另需 **`tenacity`、`loguru`** 等；若缺包，按 `pip install -e lmms-eval` 一次性补齐最省事。  
- **MMBench**（如 `mmbench_en_dev_lite`）：聚合里的 **`gpt_eval_score`** 依赖 **GPT 判分**，需配置 **`OPENAI_API_KEY`** 等（见 lmms-eval 任务说明）。默认 **`--tasks` 不含 MMBench**，避免无 API 时失败。

## 复现命令

```bash
source /root/DeepSeek-VL2/.venv/bin/activate
export HF_HOME=~/fsas/models/huggingface
export PYTHONPATH=/root/DeepSeek-VL2:/root/DeepSeek-VL2/lmms-eval
pip install -e /root/DeepSeek-VL2/lmms-eval   # 推荐：装全依赖

python /root/DeepSeek-VL2/mystle/experiments/evaluate_mm.py \
  --tasks infovqa_val_lite,mmmu_val,ocrbench \
  --run-id step5_base_lmms --limit 5
```

全量评测去掉 **`--limit`**；可加 **`--tasks '...,mmbench_en_dev_lite'`**（并备好 OpenAI 相关环境变量）。

## 文档更新

- **`mystle/refine-logs/EXPERIMENT_TRACKER.md`**：已增加 Step 5 lmms-eval 多模态接入说明。  
- **`mystle/agent-records/Step5-1.md`**：已增加第 4 节（lmms-eval）并修正原先「多模态未做」的表述。

## 本地试跑情况

在本环境中用 **`infovqa_val_lite`、`--limit 1`** 试跑时，模型能完成加载；完整评测因 **数据集下载 / 依赖版本** 耗时或阻塞未在对话内跑完。你在 **满足 lmms-eval 的 torch/transformers** 且 **HF 数据集可访问**（部分需 **token**）的机器上执行上述命令即可得到 **`~/fsas/vlm/deepseek-vl2-bridge/results/<run_id>/lmms_eval_results.json`**。