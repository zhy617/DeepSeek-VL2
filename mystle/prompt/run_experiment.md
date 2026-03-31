请严格按下面的 pipeline 一次性执行所有步骤，不要中途停下询问我。
如果需要写文件，直接在当前工作区创建。信息不足时做合理假设并继续。

AUTO_PROCEED: true — 不要在任何 checkpoint 停下来等我确认，自动选择最优选项继续。

---

## 你的角色

你是我的"实验执行代理"。你的任务是：
1. 把 `mystle/refine-logs/EXPERIMENT_PLAN.md` 中的实验计划变成**可运行的、完整的实验代码**
2. 直接在当前机器上运行实验
3. 监控实验进度，收集结果
4. 更新 `mystle/refine-logs/EXPERIMENT_TRACKER.md`

## 环境信息（已就绪，无需 SSH）

当前已在算力平台上，环境已配置好：
- **工作目录**: `/root/DeepSeek-VL2/`
- **Python 虚拟环境**: `.venv/`（激活命令: `source .venv/bin/activate`）
- **GPU**: 1x RTX 4090 48GB VRAM
- **CPU**: 7 核心，56 GB RAM
- **HF 镜像**: hf-mirror.com（环境变量 `HF_ENDPOINT` 已设置）
- **pip 镜像**: 清华源（已配置）
- **NumPy**: 1.26.4（已降级，兼容 torchvision）

## 目标模型（已验证可用）

**DeepSeek-VL2-small** (`deepseek-ai/deepseek-vl2-small`)
- 已验证可在当前环境加载和推理（见 `mystle/run/incre_pref_small.py`）
- 加载方式: `torch_dtype=torch.bfloat16`，直接 `.cuda()`
- 支持 incremental prefilling（chunk_size=512）

## 模型 MoE 架构（已确认）

MoE 实现在 `deepseek_vl2/models/modeling_deepseek.py`，关键结构：

| 类 | 作用 |
|---|------|
| `MoEGate` | 线性门控，softmax/sigmoid 打分，支持 greedy/group_limited_greedy/noaux_tc top-k 策略 |
| `DeepseekV2MoE` | MoEGate + ModuleList 的 `DeepseekV2MLP` 专家 + 可选 shared_experts |
| `DeepseekV2DecoderLayer` | 当 `layer_idx >= first_k_dense_replace` 且 `layer_idx % moe_layer_freq == 0` 时 mlp 为 MoE |

配置关键字段（`configuration_deepseek.py` → `DeepseekV2Config`）：
- `n_routed_experts`: 路由专家数量
- `num_experts_per_tok`: 每个 token 激活的专家数 (top-K)
- `moe_layer_freq`: MoE 层频率
- `first_k_dense_replace`: 前 k 层为 dense（不使用 MoE）
- `moe_intermediate_size`: 专家 FFN 中间维度

VL2 多模态模型（`modeling_deepseek_vl_v2.py`）使用 `DeepseekV2ForCausalLM` 作为语言塔，MoE 来自 decoder 层。

## 现有代码基础

```
mystle/
├── README.md
├── prompt/
│   └── run_experiment.md      ← 当前文件
├── refine-logs/
│   └── EXPERIMENT_PLAN.md     ← 实验计划（已有）
└── run/
    └── incre_pref_small.py    ← 已验证的模型加载+推理脚本
```

已有参考文档：
- `mystle/refine-logs/FINAL_PROPOSAL.md` — 完整方法描述和 novelty 论证
- `mystle/refine-logs/EXPERIMENT_PLAN.md` — 实验计划和 claim map
- `mystle/refine-logs/EXPERIMENT_TRACKER.md` — 实验进度追踪表
- `mystle/refine-logs/round-0-initial-proposal.md` — 初始提案（供参考）

需要创建的目录和文件：
- `mystle/experiments/` — 实验代码（bridge_score.py, admissibility_merge.py, evaluate.py, baselines/）
- `mystle/results/` — 实验结果输出
- `mystle/calibration_data/` — 校准数据
- `mystle/refine-logs/EXPERIMENT_RESULTS.md` — 最终结果汇总（实验完成后创建）

## 研究方法核心（必须精确实现）

### Bridge Score 计算

对每个 expert `e_i` 在每个 MoE 层 `l`，用校准数据计算：

```
I_visual(e_i) = E_x[||h_l(x) - h_l(x_visual_ablated)||]   when e_i is in top-K
I_text(e_i)   = E_x[||h_l(x) - h_l(x_text_ablated)||]     when e_i is in top-K  
I_mismatch(e_i) = E_x[||h_l(x) - h_l(x_mismatch)||]       when e_i is in top-K
```

- `x_visual_ablated`: 图像替换为全零/空白图像
- `x_text_ablated`: 文本替换为无意义 padding
- `x_mismatch`: 图像替换为同 batch 中另一条样本的图像（语义不匹配）

```
Bridge Score: B(e_i) = I_mismatch(e_i) - (I_visual(e_i) + I_text(e_i)) / 2
Modality Affinity: M(e_i) = (I_text(e_i) - I_visual(e_i)) / (I_text(e_i) + I_visual(e_i) + ε)
```

### Merge Admissibility Gate

```
A(e_i, e_j) = 1  iff  B(e_i) < τ_bridge AND B(e_j) < τ_bridge AND |M(e_i) - M(e_j)| < δ_affinity
否则 A(e_i, e_j) = 0
高 bridge experts (B > τ_bridge): frozen 不参与合并
```

## 模型加载模板

基于已验证的加载方式：

```python
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
)
vl_gpt = vl_gpt.cuda().eval()
```

## 执行步骤（严格按顺序）

### Step 0: 环境验证

1. 激活虚拟环境: `source .venv/bin/activate`
2. 检查 GPU 状态: `nvidia-smi`
3. 验证依赖: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
4. 如果缺少依赖，安装:
   ```
   pip install scipy datasets matplotlib seaborn tqdm
   ```
5. 创建必要目录:
   ```
   mkdir -p mystle/experiments/baselines mystle/results mystle/calibration_data
   ```

### Step 1: 模型结构探测

1. 加载模型，打印所有 MoE 层信息
2. 记录每层的 expert 数量、top-K 设置、MoE 层索引
3. 确认能 hook 到 `DeepseekV2MoE` 层的 expert 输出
4. 保存到 `mystle/results/model_info.json`

### Step 2: 校准数据准备

1. 从 HuggingFace datasets 下载 InfoVQA 或 LLaVA-Bench 样本
2. 构造 1000 条校准样本（含 image + text question + answer）
3. 对每条样本预生成 3 种干预版本：
   - original: 正常图文对
   - visual_ablated: 图像替换为同尺寸空白/全零张量
   - mismatch: 图像替换为另一条样本的图像
4. 保存到 `mystle/calibration_data/`

### Step 3: 实现 Bridge Score 计算

创建 `mystle/experiments/bridge_score.py`，实现：
1. 模型加载（bf16，参照模板）
2. MoE 层自动检测和 hook 注册（hook `DeepseekV2MoE` 的 forward）
3. 三种干预条件的 forward pass
4. 逐 expert 的 hidden state 差异计算
5. Bridge score 和 modality affinity 计算
6. 跨 3 个校准子集的 stability 分析
7. JSON 结果输出 + 分布可视化（matplotlib/seaborn）
8. 内存管理（batch 处理，及时 `torch.cuda.empty_cache()`）

### Step 4: 实现 HC-SMoE Baseline

创建 `mystle/experiments/baselines/hcsmoe_merge.py`：
1. 基于 expert 输出相似度的层次聚类
2. 按聚类结果合并 expert 权重（weighted average）
3. 支持不同压缩率 (25%, 50%)
4. 合并后的模型能正常推理

### Step 5: 实现评测脚本

创建 `mystle/experiments/evaluate.py`：
1. 支持 InfoVQA、OCRBench、MMMU、MMBench 的评测
2. 使用 lmms-eval 框架或手写评测逻辑
3. 输出 JSON 格式结果
4. 计算 accuracy / score retention (%)

### Step 6: 运行实验 (Block B0 — Sanity Check)

按 EXPERIMENT_PLAN.md Block B0：
1. 运行 bridge score 计算（在后台 `nohup` 或 `screen` 中运行）
2. 检查 bridge score 分布
3. 计算 rank correlation
4. **Go/No-Go 判断**: rank correlation >= 0.8?

### Step 7: 运行 Baseline + Main Experiments (B1)

如果 B0 通过：
1. 运行 HC-SMoE baseline (25%, 50%)
2. 运行 MergeMoE baseline (25%, 50%)  
3. 运行 admissibility-gated merge (25%, 50%, 3 seeds)
4. 评测所有配置
5. 更新 EXPERIMENT_TRACKER.md

### Step 8: 运行 Novelty Isolation + Router Independence (B3, B4)

1. B3: bridge-score vs routing-freq vs activation-sim vs random vs no-protection
2. B4: naive merge + Router KD vs ours (no KD) vs ours + KD
3. 评测并记录结果

### Step 9: 结果收集与报告

1. 所有结果保存到 `mystle/results/`
2. 生成结果汇总表
3. 更新 `mystle/refine-logs/EXPERIMENT_TRACKER.md` 状态
4. 写 `mystle/refine-logs/EXPERIMENT_RESULTS.md` 汇总报告

## 代码质量要求

- 所有 hyperparameters 必须通过 argparse 可配置
- 固定 random seed，默认 42
- 结果保存为 JSON，每次运行带时间戳
- 日志用 `tee` 同时输出到文件
- 内存敏感：batch size 从小开始，遇 OOM 自动减半重试
- 长时间任务用 `nohup` 在后台运行，输出重定向到日志文件

## 实验管理

- 日志文件: `mystle/results/<run_id>.log`
- 结果文件: `mystle/results/<run_id>/`
- 每个实验完成后立即更新 `mystle/refine-logs/EXPERIMENT_TRACKER.md`

## 遇到问题时的处理

- **OOM**: 减小 batch size → 用量化 (int8/int4) → 减少校准样本数
- **模型下载慢**: 确认用了 hf-mirror.com (`export HF_ENDPOINT=https://hf-mirror.com`)
- **MoE 结构不明**: 参照上方架构表，或 `model.named_modules()` 定位
- **评测框架问题**: 改用简单的直接推理 + 正则匹配评测
- **任何阻塞**: 记录问题，跳过该步骤继续下一个，不要停下来问我

## 最终产出

在聊天末尾输出：
1. 已完成的实验列表（含状态）
2. 关键数字（如果已产出）
3. 遇到的问题和解决方案
4. 下一步建议
