#!/usr/bin/env python3
"""
Step 5：使用 lmms-eval 对 DeepSeek-VL2 **基座** 做多模态榜单评测。

默认任务（与 ``run_experiment.md`` 对齐，可按需删减）：
  - ``infovqa_val_lite``：LMMs-Eval-Lite 子集，ANLS
  - ``mmmu_val``：MMMU validation
  - ``ocrbench``：OCRBench
  - ``mmbench_en_dev_lite``（可选）：需配置 ``OPENAI_API_KEY`` 等，聚合阶段用 GPT 判分

用法::

    export HF_HOME=~/fsas/models/huggingface
    export PYTHONPATH=/root/DeepSeek-VL2:/root/DeepSeek-VL2/lmms-eval
    python mystle/experiments/evaluate_mm.py --run-id step5_base_mm --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_LMMS = _REPO_ROOT / "lmms-eval"
if _LMMS.is_dir() and str(_LMMS) not in sys.path:
    sys.path.insert(0, str(_LMMS))


def _default_results_dir() -> Path:
    return Path(os.environ.get("HOME", "/root")) / "fsas/vlm/deepseek-vl2-bridge/results"


def _register_deepseek_vl2_model() -> None:
    import mystle.experiments.deepseek_vl2_lmms  # noqa: F401 — 触发 @register_model

    from lmms_eval.models import MODEL_REGISTRY_V2
    from lmms_eval.models.registry_v2 import ModelManifest

    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(
            model_id="deepseek_vl2",
            simple_class_path="mystle.experiments.deepseek_vl2_lmms.DeepseekVL2LMMS",
        ),
        overwrite=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="lmms-eval 多模态评测 DeepSeek-VL2 基座")
    p.add_argument("--pretrained", default="deepseek-ai/deepseek-vl2-small")
    p.add_argument(
        "--tasks",
        default="infovqa_val_lite,mmmu_val,ocrbench",
        help="逗号分隔任务名；可加 mmbench_en_dev_lite（需 OpenAI 判分环境）",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--limit", type=float, default=None, help="每任务样本上限；<1 表示比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunk-size", type=int, default=512, help="incremental_prefilling 块大小，OOM 可减小")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help="bootstrap 次数；0 不算 stderr，加快结束",
    )
    p.add_argument(
        "--save-full",
        action="store_true",
        help="写入完整结果（含 configs，体积可能很大）",
    )
    args = p.parse_args()

    _register_deepseek_vl2_model()

    from lmms_eval.evaluator import simple_evaluate
    from lmms_eval.utils import handle_non_serializable

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # device / batch_size 由 simple_evaluate 经 additional_config 传入，勿在 model_args 中重复
    model_args = (
        f"pretrained={args.pretrained},"
        f"dtype={args.dtype},"
        f"chunk_size={args.chunk_size}"
    )

    results_root = args.results_root or _default_results_dir()
    run_id = args.run_id or f"step5_mm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = results_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results = simple_evaluate(
        model="deepseek_vl2",
        model_args=model_args,
        tasks=task_list,
        batch_size=args.batch_size,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        bootstrap_iters=args.bootstrap_iters,
    )

    summary_path = out_dir / "lmms_eval_results.json"
    keys = ("results", "versions", "group_subtasks")
    if args.save_full:
        keys = (*keys, "configs")
    compact = {k: results[k] for k in keys if k in results}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False, default=handle_non_serializable)

    meta = {
        "run_id": run_id,
        "pretrained": args.pretrained,
        "tasks": task_list,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "model_type": "deepseek_vl2",
        "framework": "lmms-eval",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_json": str(summary_path),
        "save_full": args.save_full,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Done. Results: {summary_path}")


if __name__ == "__main__":
    main()
