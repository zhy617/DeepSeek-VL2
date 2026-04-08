# 算力平台复用备忘

适用场景：后续在同一算力平台上跑其他项目时，作为环境与排障参考。

这份备忘只总结已经验证过的“平台使用方式”和“常见排障手法”。其中 `~/fsas`、家目录结构、基础 shell 用法大概率稳定；GPU 型号、CUDA 版本、PyTorch/transformers 组合、是否需要补丁则可能随机器或新项目而变化。

## 1. 当前平台上相对稳定的约定

- 工作目录常见为：`/root/<repo-name>`
- 项目一般使用仓库内虚拟环境：`.venv/`
- 共享持久化存储：`~/fsas`
- `~/fsas` 关机不丢，适合放数据集、模型缓存、评测结果、日志
- 系统盘只放代码和少量脚本，不要把大结果直接写进 git 仓库
- HuggingFace 镜像通常可用：`https://hf-mirror.com`
- pip 镜像通常可配清华源

## 2. `~/fsas` 目录约定

平台上已经反复出现的结构：

- `~/fsas/datasets/`：数据集、本地处理后的样本、`HF_DATASETS_CACHE`
- `~/fsas/models/`：模型权重缓存；常把 `HF_HOME` 指到 `~/fsas/models/huggingface`
- `~/fsas/pip-cache/`：pip 缓存
- `~/fsas/projects/`：按项目名归档的结果、日志、中间产物

对全新的独立项目，建议按“项目名”单独开子目录，不要复用旧项目的结果目录，例如：

```bash
export PROJECT_SLUG=my-project
export HF_HOME="${HF_HOME:-$HOME/fsas/models/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/fsas/datasets/$PROJECT_SLUG/hf_datasets_cache}"
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fsas/projects/$PROJECT_SLUG}"
export RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
export LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$RESULTS_DIR" "$LOG_DIR"
```

## 3. 建议每次开工先做的探测

不要假设“同一平台上的新项目环境可以直接照搬旧项目”，先跑这组检查：

```bash
cd /root/<repo-name>
source .venv/bin/activate

pwd
nvidia-smi
python -V
which python
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available())"
python -c "import transformers, numpy; print('transformers=', transformers.__version__); print('numpy=', numpy.__version__)"
```

额外建议记录：

- GPU 名称
- 显存大小
- `torch.__version__`
- `torch.version.cuda`
- `torch.cuda.is_available()`
- `transformers.__version__`
- `numpy.__version__`

## 4. 当前平台上通用性较强的环境变量

适合大多数 HuggingFace 类项目；`LMMS_EVAL_DATASETS_CACHE` 和 `PYTHONPATH` 只在对应项目需要时再加：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/fsas/models/huggingface
export HF_DATASETS_CACHE=~/fsas/datasets/<project-slug>/hf_datasets_cache
```

如果项目包含额外评测子模块，再补：

```bash
export LMMS_EVAL_DATASETS_CACHE="$HF_DATASETS_CACHE"
export PYTHONPATH=/root/<repo-name>:/root/<repo-name>/lmms-eval
```

如果任务需要 token，可再加：

```bash
export HF_TOKEN="$(tr -d ' \n\r' < /root/hf_token.txt)"
```

## 5. 这个平台上已经踩过的坑

下面这些坑来自当前仓库的真实经历，但对后续其他项目也有参考价值。

### 5.1 GPU 可见，但 PyTorch 不能用 CUDA

已出现过：

- `nvidia-smi` 正常
- 但 `torch.cuda.is_available()` 返回 `False`

已知原因：

- 当前 `.venv` 里的 PyTorch CUDA 构建版本，和机器实际驱动/运行时不匹配
- 例如安装的是更高 CUDA 版本对应的 PyTorch，而当前机器只暴露较低 CUDA 运行时

处理思路：

1. 先看 `nvidia-smi`
2. 再看 `torch.version.cuda`
3. 让 PyTorch 安装版本与当前机器 CUDA 运行时匹配
4. 重验 `torch.cuda.is_available()`

结论：在同一平台启动一个新项目时，最需要优先确认的是 “驱动/CUDA/PyTorch 三者是否对齐”。

### 5.2 `xformers` 很容易在新项目或新环境中失配

已出现过：

- `xformers.ops.memory_efficient_attention` 在旧环境能用
- 项目环境变化后，`xformers` 仍在，但 CUDA 扩展不可调度
- 前向时报 `NotImplementedError`

处理思路：

1. 不要默认 `xformers` 可用
2. 如果模型代码强依赖 `memory_efficient_attention`，要准备 fallback
3. 常见 fallback 是 `torch.nn.functional.scaled_dot_product_attention`

给另一个 agent 的建议：

- 看到 attention 算子报错时，先怀疑 `xformers` 和 torch/cu 失配
- 优先做最小兼容补丁，不要一上来大改模型逻辑

### 5.3 额外子模块常有独立依赖

常见情况：

- 主项目能跑，不代表附带的评测/训练/服务子模块也能直接跑
- 子模块可能对 `torch` / `transformers` / `cuda` 有单独要求

实用做法：

```bash
pip install -e ./some-submodule
```

然后确认：

- `PYTHONPATH` 包含仓库根目录
- `PYTHONPATH` 包含对应子模块目录
- 子模块自己的依赖说明已经看过一遍

## 6. 存储与日志建议

适合复用到别的项目：

- 每个新项目单独使用一个 `PROJECT_ROOT`，例如 `~/fsas/projects/<project-slug>/`
- 结果写到 `"$PROJECT_ROOT/results/<run_id>/"` 
- 日志写到 `"$PROJECT_ROOT/logs/<run_id>.log"`
- 长任务用 `nohup` 或类似后台方式运行
- 重要运行都保存 `meta.json`，至少记：
  - 命令
  - 时间
  - 环境变量
  - 依赖版本
  - GPU 信息
  - 输入/输出路径

一个通用示例：

```bash
export RUN_ID=test_run_001
nohup python path/to/run.py \
  --output-dir "$RESULTS_DIR/$RUN_ID" \
  > "$LOG_DIR/$RUN_ID.log" 2>&1 &
```

## 7. 新项目启动模板

如果要在这个平台上启动一个新项目，可以先按下面流程执行：

```bash
cd /root/<repo-name>

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-$HOME/fsas/models/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/fsas/datasets/<project-slug>/hf_datasets_cache}"
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/fsas/projects/<project-slug>}"
export RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
export LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$PROJECT_ROOT" "$RESULTS_DIR" "$LOG_DIR"

nvidia-smi
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import transformers, numpy; print(transformers.__version__, numpy.__version__)"
```

然后按结果分支判断：

- 若 `torch.cuda.is_available()` 为 `False`：先修环境，不要直接开跑
- 若模型前向里有 `xformers` 报错：优先做 attention fallback
- 若需要 HF 数据集/权重：确认缓存走 `~/fsas`，避免把系统盘写满
- 若需要评测子模块：先检查是否要 `pip install -e` 与补 `PYTHONPATH`

## 8. 一句话原则

这个平台最值得复用的不是“某个固定版本号”，而是这套工作方式：

- 代码在仓库
- 每个新项目在 `~/fsas/projects/<project-slug>/` 单独留痕
- 共享缓存和数据放 `~/fsas`
- 每次先探测 GPU/torch/CUDA
- 长任务后台跑并落日志
- 出现环境兼容问题时，优先做最小恢复，再继续实验
