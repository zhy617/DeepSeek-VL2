#!/usr/bin/env python3
"""
Step 5：使用仓库内 ``lm-evaluation-harness`` 评测 DeepSeek-VL2 **基座**（未合并 checkpoint）。

说明：
- **InfoVQA / OCRBench / MMMU / MMBench** 为多模态榜单，需 ``lmms-eval`` 或专用管线；本脚本用 lm-eval 的 **纯文本任务**
  作为语言骨干能力的可复现基准（与 ``run_experiment.md`` 中「或手写 / lmms-eval」一致）。
- 基座模型 ID 默认 ``deepseek-ai/deepseek-vl2-small``（与项目其余实验一致）。

用法示例::

    export HF_HOME=~/fsas/models/huggingface
    export PYTHONPATH=/root/DeepSeek-VL2
    python mystle/experiments/evaluate.py --run-id step5_base_lm_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# 仓库根目录需在 path 中（deepseek_vl2 与 mystle）
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _default_results_dir() -> Path:
    return Path(os.environ.get("HOME", "/root")) / "fsas/vlm/deepseek-vl2-bridge/results"


def main() -> None:
    p = argparse.ArgumentParser(description="lm-eval 评测 DeepSeek-VL2 基座（文本任务）")
    p.add_argument(
        "--pretrained",
        default="deepseek-ai/deepseek-vl2-small",
        help="HF 模型 ID 或本地目录",
    )
    p.add_argument(
        "--tasks",
        default="arc_easy,winogrande,piqa",
        help="逗号分隔的 lm-eval 任务名",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="仅测前 N 条（调试用）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--run-id",
        default=None,
        help="结果子目录名；默认带时间戳",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="结果根目录，默认 ~/fsas/vlm/deepseek-vl2-bridge/results",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--save-full",
        action="store_true",
        help="同时写入完整 lm-eval 输出（含 configs，体积可能很大）",
    )
    args = p.parse_args()

    # 注册 VL2 + 自定义 HFLM（必须在 import lm_eval evaluate 之前）
    import mystle.experiments.deepseek_vl2_lm_eval  # noqa: F401

    from lm_eval import simple_evaluate
    from lm_eval.utils import handle_non_serializable

    results_root = args.results_root or _default_results_dir()
    run_id = args.run_id or f"step5_base_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = results_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    model_args = (
        f"pretrained={args.pretrained},"
        "trust_remote_code=True,"
        "dtype=bfloat16"
    )

    results = simple_evaluate(
        model="deepseek-vl2-hf",
        model_args=model_args,
        tasks=task_list,
        batch_size=args.batch_size,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        device=args.device,
    )

    summary_path = out_dir / "lm_eval_results.json"
    compact = {
        k: results[k]
        for k in ("results", "versions", "group_subtasks")
        if k in results
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False, default=handle_non_serializable)
    if args.save_full:
        full_path = out_dir / "lm_eval_full_dump.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(
                results, f, indent=2, ensure_ascii=False, default=handle_non_serializable
            )

    meta = {
        "run_id": run_id,
        "pretrained": args.pretrained,
        "tasks": task_list,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "model_type": "deepseek-vl2-hf",
        "framework": "lm-evaluation-harness",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_json": str(summary_path),
        "save_full": args.save_full,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Done. Results: {summary_path}")


if __name__ == "__main__":
    main()
