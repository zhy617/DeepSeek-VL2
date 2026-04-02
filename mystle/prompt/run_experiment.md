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

## 共享存储 `~/fsas`（数据集与实验结果）

算力平台共享盘，**关机不丢**；数据集与实验产出放在此处，避免占满系统盘。

实际顶层结构（`~/fsas` 下）：

| 目录 | 用途 |
|------|------|
| `datasets/` | 数据集：校准样本、评测集缓存、HuggingFace `datasets` 库缓存等 |
| `models/` | 模型权重缓存；`HF_HOME` 指向 `~/fsas/models/huggingface`（见 `mystle/README.md`） |
| `pip-cache/` | pip 缓存，勿写入实验数据 |
| `vlm/` | 多模态/VLM 相关实验产出：日志、JSON、图、中间检查点 |

**本实验建议路径约定**（在以上目录下建子目录，避免与其它任务混放）：

```
~/fsas/datasets/deepseek-vl2-bridge/
├── calibration/          # 1000 条校准样本及预处理
└── hf_datasets_cache/    # 可选：export HF_DATASETS_CACHE=... 指向此处

~/fsas/vlm/deepseek-vl2-bridge/
├── results/              # 各 run 的 JSON、图
├── logs/                 # nohup / tee 日志
└── model_info.json       # Step 1 结构探测（或放在 results/ 下）
```

**环境变量（在 shell 或脚本开头设置）**：

```bash
export HF_HOME="${HF_HOME:-$HOME/fsas/models/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/fsas/datasets/deepseek-vl2-bridge/hf_datasets_cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HOME/fsas/datasets/deepseek-vl2-bridge/calibration" \
  "$HOME/fsas/vlm/deepseek-vl2-bridge/results" "$HOME/fsas/vlm/deepseek-vl2-bridge/logs"
```

代码仓库内仍只放 **`mystle/experiments/`** 源码；**不要**把大体积数据或结果提交进 git。

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
- `~/fsas/datasets/deepseek-vl2-bridge/calibration/` — 校准数据（大文件，在共享盘）
- `~/fsas/vlm/deepseek-vl2-bridge/` — 实验结果、日志（`results/`、`logs/` 子目录）
- `mystle/refine-logs/EXPERIMENT_RESULTS.md` — 最终结果文字汇总（实验完成后创建，可引用 fsas 中的路径）

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

**说明**：B 可为正或负；叙事与门控应依赖**相对排序 / 层内标准化**，不要求全局 B>0（见下「实证」）。

### 实证结论（DeepSeek-VL2-small，1000 条校准，`bridge_score_1000_full`）

- **层间差异主导**：各 MoE 层上 **pooled B** 的层均值 \(\bar{B}_l\) 跨度大；**层内**跨 expert 的标准差 \(\sigma_l\) 在多数层较小，在少数层（如高 \(|\bar{B}_l|\) 或高 \(\sigma_l\) 的层）明显更大。
- **推论**：原「仅按全局阈值 \(\tau_{\mathrm{bridge}}\) 区分每个 expert」与数据不符；应改为 **层优先（layer-first）**，仅在 **高离散层** 上保留 **专家级（expert-level）** 精细门控。

### Layer-First + Expert-Refined Admissibility（默认实现目标）

对每层 \(l\) 先算（由 `bridge_score_results.json` 的 `pooled_all` 可得）：

- **层均值**：\(\bar{B}_l = \frac{1}{N}\sum_{e} B(e,l)\)
- **层内离散度**：\(\sigma_l = \mathrm{std}_e B(e,l)\)
- **层内 z 分数**（用于同层比较）：\(B_z(e,l) = \dfrac{B(e,l) - \bar{B}_l}{\sigma_l + \varepsilon}\)

**(1) 层策略（merge 预算 / 保护强度）**

- 用 \(\bar{B}_l\) 在所有 MoE 层上的**分位数**或 z-score 定义 **层敏感度** \(S_l\)（具体映射在 `admissibility_merge.py` 中用 argparse 固定一种，并在论文中报告）。
- **高敏感度层**（例如 \(S_l\) 高于某分位）：**降低该层可合并 expert 比例**、或**提高该层冻结比例**；**低敏感度层**可接近基线 HC-SMoE 的合并强度。

**(2) 专家策略（条件于 \(\sigma_l\)）**

- 若 \(\sigma_l < \tau_{\mathrm{disp}}\)（层内几乎「一条横带」）：**不在该层做精细 per-expert bridge 排序**；合并仅在 **输出相似度**（或路由共现）下决定，门控主要由 **(1) 的层预算**约束。
- 若 \(\sigma_l \geq \tau_{\mathrm{disp}}\)（层内 expert 差异大）：启用 **per-expert** 规则——用 **\(B_z(e,l)\)** 与 \(M(e,l)\) 定义可合并对与冻结集（见下）。

**(3) 可合并对与冻结（高离散层用 \(B_z\)，避免全局绝对阈值）**

```
A(e_i, e_j) = 1  iff  同层 l 且 σ_l ≥ τ_disp
              且  B_z(e_i,l) < τ_z 且 B_z(e_j,l) < τ_z
              且  |M(e_i) - M(e_j)| < δ_affinity
否则（或 σ_l < τ_disp 时由层预算 + 相似度聚类决定）按实现分支处理。

高 bridge（相对同层）：B_z(e,l) > τ_z 的 expert → 该层内冻结或降低合并优先级（τ_z 取同层分位数，而非全局常数）。
```

**超参**：\(\tau_{\mathrm{disp}}\)、\(\tau_z\)、\(\delta_{\mathrm{affinity}}\)、层敏感度分位与 **总压缩率** 一起在实现中可配置；小验证集上网格搜索或固定为论文主表一组 + 附录敏感性分析。

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
5. 创建必要目录（含共享盘上的数据与结果路径，见上文「共享存储」）:
   ```
   mkdir -p mystle/experiments/baselines \
     "$HOME/fsas/datasets/deepseek-vl2-bridge/calibration" \
     "$HOME/fsas/datasets/deepseek-vl2-bridge/hf_datasets_cache" \
     "$HOME/fsas/vlm/deepseek-vl2-bridge/results" \
     "$HOME/fsas/vlm/deepseek-vl2-bridge/logs"
   ```

### Step 1: 模型结构探测

1. 加载模型，打印所有 MoE 层信息
2. 记录每层的 expert 数量、top-K 设置、MoE 层索引
3. 确认能 hook 到 `DeepseekV2MoE` 层的 expert 输出
4. 保存到 `~/fsas/vlm/deepseek-vl2-bridge/results/model_info.json`（或同目录下带时间戳子文件夹）

### Step 2: 校准数据准备

1. 从 HuggingFace datasets 下载 InfoVQA 或 LLaVA-Bench 样本
2. 构造 1000 条校准样本（含 image + text question + answer）
3. 对每条样本预生成 3 种干预版本：
   - original: 正常图文对
   - visual_ablated: 图像替换为同尺寸空白/全零张量
   - mismatch: 图像替换为另一条样本的图像
4. 保存到 `~/fsas/datasets/deepseek-vl2-bridge/calibration/`

### Step 3: 实现 Bridge Score 计算

创建 `mystle/experiments/bridge_score.py`，实现：
1. 模型加载（bf16，参照模板）
2. MoE 层自动检测和 hook 注册（hook `DeepseekV2MoE` 的 forward）
3. 三种干预条件的 forward pass
4. 逐 expert 的 hidden state 差异计算
5. Bridge score 和 modality affinity 计算
6. 跨 3 个校准子集的 stability 分析
7. JSON 结果输出 + 分布可视化（matplotlib/seaborn），默认写入 `~/fsas/vlm/deepseek-vl2-bridge/results/`
8. 内存管理（batch 处理，及时 `torch.cuda.empty_cache()`）
9. **（供 Step 4）** 在 JSON 中已有每层 `pooled_all`；实现或脚本中导出 **层汇总表**（\(\bar{B}_l\)、\(\sigma_l\)、可选 \(\tau_{\mathrm{disp}}\) 标记）到同目录 `layer_bridge_summary.json`，供 `admissibility_merge.py` 读取

### Step 4: 实现 HC-SMoE Baseline + Admissibility-Gated Merge（Layer-First）

创建 `mystle/experiments/baselines/hcsmoe_merge.py`（及 `mystle/experiments/admissibility_merge.py`）：
1. 基于 expert 输出相似度的层次聚类
2. 按聚类结果合并 expert 权重（weighted average）
3. 支持不同压缩率 (25%, 50%)
4. 合并后的模型能正常推理
5. **Ours**：读取 `bridge_score_results.json` + `layer_bridge_summary.json`，按上文 **Layer-First + Expert-Refined** 规则约束每层合并预算与高 \(\sigma_l\) 层上的 \(B_z\) 冻结集

### Step 5: 实现评测脚本

创建 `mystle/experiments/evaluate.py`：
1. 支持 InfoVQA、OCRBench、MMMU、MMBench 的评测
2. 使用 lmms-eval 框架或手写评测逻辑
3. 输出 JSON 格式结果到 `~/fsas/vlm/deepseek-vl2-bridge/results/`
4. 计算 accuracy / score retention (%)

### Step 6: 运行实验 (Block B0 — Sanity Check)

按 EXPERIMENT_PLAN.md Block B0：
1. 运行 bridge score 计算（在后台 `nohup` 或 `screen` 中运行）
2. 检查 **全局与层间** B 分布、\(\bar{B}_l\) 与 \(\sigma_l\)（热力图 + `layer_bridge_summary.json`）
3. 计算跨子集 rank correlation（Spearman）
4. **Go/No-Go**：子集 Spearman \(\geq 0.8\) **且** 层间 \(\bar{B}_l\) 分层可解释（非退化常数）；不要求「全局高 B expert 子集」或「全体 B>0」

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

1. 所有结果保存到 `~/fsas/vlm/deepseek-vl2-bridge/results/`（按 `run_id` 分子目录）
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

- 日志文件: `~/fsas/vlm/deepseek-vl2-bridge/logs/<run_id>.log`
- 结果文件: `~/fsas/vlm/deepseek-vl2-bridge/results/<run_id>/`
- 每个实验完成后立即更新 `mystle/refine-logs/EXPERIMENT_TRACKER.md`（Notes 中可写 fsas 路径）

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
