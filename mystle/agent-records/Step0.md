
根据 `run_experiment.md` 与 `EXPERIMENT_PLAN.md`，实验涉及的**原始 HuggingFace 数据**如下；`~/fsas/datasets/deepseek-vl2-bridge/calibration/` 里是 Step 2 **加工后的 1000 条校准集**，不是 Hub 直下的原始包，目前仍为空（需后续用脚本从下列数据采样生成）。

## 已有缓存（无需重复下载）

| 用途 | 数据集 ID | 说明 |
|------|-----------|------|
| 校准采样 + 评测 | `Ahren09/info_vqa` | train/val/test 分片已在缓存（约 11GB） |
| 评测 | `echo840/OCRBench` | `test` 1000 条 |
| 校准采样 | `lmms-lab/llava-bench-in-the-wild` | `train` 60 条 |
| 评测 | `lmms-lab/MMBench` | 需 **`config_name='en'`**，`dev` 4329 条（计划从中抽 1000） |
| 评测 | `MMMU/MMMU` | **23/30** 个学科子集已在缓存 |

## 本次补全（之前未下全的部分）

- **`MMMU/MMMU`** 在共享盘里只有 **23** 个学科目录，缺 **7** 个：`Mechanical_Engineering`, `Music`, `Pharmacy`, `Physics`, `Psychology`, `Public_Health`, `Sociology`。  
- 已用 `load_dataset("MMMU/MMMU", <subject>, split="validation")` 拉齐；每个学科 **validation = 30 条**，与原有学科一致；现在 **`MMMU___mmmu/` 下共 30 个子目录**。

（下载过程中曾对 `cas-bridge.xethub.hf.co` 出现一次 **Read timeout**，库自动重试后成功；`MMBench en dev` 在超时后也从缓存正常生成。）

## 未重复下载的说明

- **MMBench `cc` / `cn`**：计划写的是 **MMBench dev (1000)**，与英文评测一致，只保证 **`en`**；若你将来要中文/繁体榜，再单独下对应 config。
- **DeepSeek-VL2 模型权重**：在 `HF_HOME`（`~/fsas/models/huggingface`），不在 `datasets/`；本次未动。
- **校准目录 `calibration/`**：仍为空属预期；需在 Step 2 从 **InfoVQA train + LLaVA-Bench** 采样并写盘，不是“再下一个 Hub 数据集”。

## 使用时的 split 提示（避免写评测脚本时踩坑）

- **InfoVQA**：Hub 上是 **`val`**，没有名为 `dev` 的 split；计划里 “dev 500” 应对 **`val` 子采样**。
- **MMMU**：多学科是 **30 个 `config_name`**；**`validation`** 合计 900 条（30×30），与计划 “MMMU dev 900” 一致；若用 **`dev`** 则是每学科 5 条、合计 150，与计划不符，评测应对齐 **`validation`**。

若你希望，我可以加一个**纯离线**的 `mystle/scripts/verify_datasets.py`（用缓存路径检查，不访问 Hub），方便以后一键检查是否缺文件。