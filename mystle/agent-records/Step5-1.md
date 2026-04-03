

## 已完成内容

### 1. Step 5 与 `lm-evaluation-harness` 的衔接
- **`mystle/experiments/deepseek_vl2_lm_eval.py`**  
  - 向 Transformers 注册 `deepseek_vl_v2` → `DeepseekVLV2Config` / `DeepseekVLV2ForCausalLM`（Hub 快照无 `.py`，必须用本仓库 `deepseek_vl2`）。  
  - 注册 lm-eval 模型 **`deepseek-vl2-hf`**：在 `HFLM._create_model` 期间临时包装 `from_pretrained`，去掉值为 `None` 的 `gguf_file` / `quantization_config`（否则会报 unexpected keyword）。  
  - 加载后为顶层 `DeepseekVLV2Config` 补全 `use_cache` 等字段（否则 forward 读 `self.config.use_cache` 会报错）。

### 2. **`mystle/experiments/evaluate.py`**
- 调用 `simple_evaluate(..., model="deepseek-vl2-hf", ...)`，默认任务 **`arc_easy,winogrande,piqa`**，结果写入 `~/fsas/vlm/deepseek-vl2-bridge/results/<run_id>/`。  
- 默认只保存 **`results` / `versions` / `group_subtasks`**，避免把 `configs` 打进 JSON 造成巨大文件；需要完整对象时加 **`--save-full`**。

### 3. **基座模型全量评测（已跑完）**
- **`run_id`**: `step5_base_lm_eval_full`  
- **模型**: `deepseek-ai/deepseek-vl2-small`，bf16，`batch_size=2`，`seed=42`  
- **指标（0-shot，acc）**:
  - **arc_easy**: **77.86%**（test 2376）  
  - **winogrande**: **70.24%**（validation 1267）  
  - **piqa**: **77.53%**（validation 1838；acc_norm 约 78.4%）  

**产出路径**  
- 结果：`/root/fsas/vlm/deepseek-vl2-bridge/results/step5_base_lm_eval_full/lm_eval_results.json`  
- 元数据：`.../meta.json`  
- 日志：`/root/fsas/vlm/deepseek-vl2-bridge/logs/step5_base_lm_eval_full.log`  

已把 **`mystle/refine-logs/EXPERIMENT_TRACKER.md`** 更新为包含上述 Step 5 说明。

### 说明（与 `run_experiment.md` 对齐）
- **纯文本**基座能力见 **`evaluate.py` + lm-eval**；**InfoVQA / OCRBench / MMMU /（可选）MMBench** 等多模态见 **`evaluate_mm.py` + lmms-eval**（见上文第 4 节）。  
- **合并模型 baseline / retention** 你说过先不做，当前未评测 `merged_model/`。

### 复现命令
```bash
source /root/DeepSeek-VL2/.venv/bin/activate
export HF_HOME=~/fsas/models/huggingface
export PYTHONPATH=/root/DeepSeek-VL2
python /root/DeepSeek-VL2/mystle/experiments/evaluate.py \
  --tasks arc_easy,winogrande,piqa --batch-size 2 --run-id <新 run_id>
```

### 问题与处理
| 问题 | 处理 |
|------|------|
| `AutoConfig` 不认 `deepseek_vl_v2` | 本地注册 + `PYTHONPATH` 指向仓库 |
| `gguf_file` 传入自定义模型 | 包装 `from_pretrained` 去掉 `None` 参数 |
| 顶层 config 缺 `use_cache` | 加载后 `setattr` 默认值 |
| JSON 含 `configs` 体积极大 | 默认只写精简字段；已把本次 `lm_eval_results.json` 压成精简版 |

### 下一步（若你需要）
- 对 **merged checkpoint** 用同一 `evaluate.py` 把 `--pretrained` 指到 `merged_model/` 目录，可算 **retention %**（相对本次基座数字）。  
- 多模态榜单再接入 **lmms-eval** 或项目内生成脚本。

---

### 4. Step 5 与 **lmms-eval**（多模态）

- **`mystle/experiments/deepseek_vl2_lmms.py`**  
  - 实现 lmms-eval 的 **`lmms`** 子类 **`DeepseekVL2LMMS`**（`@register_model("deepseek_vl2")`），并注册 HF `deepseek_vl_v2`、与 lm-eval 相同的 `from_pretrained` 去 `None` 关键字、顶层 `use_cache` 修补。  
  - **`generate_until`**：按任务提供的 `doc_to_visual` 得到 PIL，将无 `<image>` 的 prompt 前缀为与图像数一致的 `<image>`，再经 `DeepseekVLV2Processor` → **`incremental_prefilling`（默认 `chunk_size=512`）** + **`generate`**，支持任务 yaml 里的 `until` / `max_new_tokens` 等。  
  - `loglikelihood` / `generate_until_multi_round`：未实现（与若干内置模型一致，当前默认任务为 `generate_until`）。

- **`mystle/experiments/evaluate_mm.py`**  
  - 将 **`lmms-eval`** 子目录加入 `sys.path`，向 **`MODEL_REGISTRY_V2`** 注册 **`deepseek_vl2`** → 上述类路径。  
  - 调用 **`lmms_eval.evaluator.simple_evaluate`**，默认任务 **`infovqa_val_lite,mmmu_val,ocrbench`**（与 `run_experiment.md` 中多模态列表对齐；**`mmbench_en_dev_lite`** 需自行加入 `--tasks`，且 **MMBench 聚合依赖 GPT 判分 API**）。  
  - 结果写入 **`~/fsas/vlm/deepseek-vl2-bridge/results/<run_id>/`**（`lmms_eval_results.json`、`meta.json`）；**`--save-full`** 才写入 `configs`。

**复现（多模态）**
```bash
source /root/DeepSeek-VL2/.venv/bin/activate
export HF_HOME=~/fsas/models/huggingface
export PYTHONPATH=/root/DeepSeek-VL2:/root/DeepSeek-VL2/lmms-eval
# 建议：pip install -e /root/DeepSeek-VL2/lmms-eval   # 安装 pyproject 依赖（torch>=2.1、transformers>=4.39 等）
python /root/DeepSeek-VL2/mystle/experiments/evaluate_mm.py \
  --tasks infovqa_val_lite,mmmu_val,ocrbench \
  --run-id step5_base_lmms --limit 5  # 调试用；全量去掉 --limit
```

**说明**  
- 部分 HF 数据集需 **token / 网络**；**MMBench** 的 **`gpt_eval_score`** 需配置 **`OPENAI_API_KEY`**（及 `API_TYPE` 等，见 `lmms-eval` 任务说明）。  
- 若与已安装的 `lmms-eval 0.7.1` 报 **torch/transformers 版本** 冲突，请以 **`lmms-eval/pyproject.toml`** 为准升级环境后再跑。