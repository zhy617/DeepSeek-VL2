### Install
#### Env

```shell
grep -qxF 'export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple' ~/.bashrc || echo 'export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple' >> ~/.bashrc

grep -qxF 'export HF_HOME=$HOME/fsas/models/huggingface' ~/.bashrc || echo 'export HF_HOME=$HOME/fsas/models/huggingface' >> ~/.bashrc

grep -qxF 'mkdir -p "$HF_HOME"' ~/.bashrc || echo 'mkdir -p "$HF_HOME"' >> ~/.bashrc

grep -qxF 'export HF_ENDPOINT=https://hf-mirror.com' ~/.bashrc || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

#### 共享盘 `~/fsas` 目录说明

平台提供的共享存储根目录（`ls ~/fsas` 常见为）：

| 目录 | 用途 |
|------|------|
| `datasets/` | 数据集、校准数据、可选 `HF_DATASETS_CACHE` |
| `models/` | `HF_HOME` → `huggingface/` 模型权重缓存 |
| `pip-cache/` | pip 缓存 |
| `vlm/` | 多模态实验结果、日志（本仓库 bridge 实验见 `mystle/prompt/run_experiment.md` 中的路径约定） |

详细子路径约定见 `mystle/prompt/run_experiment.md` 中「共享存储」一节。


#### venv
```shell
apt-get update

apt install python3.10-venv

python3 -m venv .venv

source .venv/bin/activate
```
#### pip
```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

#### VPN
安装 sing-box
```shell
curl -fsSL https://sing-box.app/install.sh | sh
```

在算力平台上，无法直接使用 `systemctl` 命令，故只能用 mix in，而且只能用 local dns，运行只能让 sing-box 自己跑
```shell
sing-box run -c /path/to/config
```

添加代理
```shell
grep -qxF 'export http_proxy=http://127.0.0.1:20122' ~/.bashrc || echo 'http_proxy=http://127.0.0.1:20122' >> ~/.bashrc

grep -qxF 'export https_proxy=https://127.0.0.1:20122' ~/.bashrc || echo 'https_proxy=https://127.0.0.1:20122' >> ~/.bashrc
```

#### debug
查找当前 python 版本，当前 python 是否支持某个版本
```shell
lsb_release -a
python3 --version
which python3
python3 -m pip --version
python3 -m pip index versions torch
python3 -m pip install "torch==2.0.1"
```