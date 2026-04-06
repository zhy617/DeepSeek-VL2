#!/usr/bin/env python3
"""
Step 5：DeepSeek-VL2 测评脚本（统一入口）。

**后端**

- ``text``：``lm-evaluation-harness``，评测语言骨干（纯文本任务）。
- ``mm``：``lmms-eval``，多模态榜单（默认与 ``run_experiment.md`` / Step5-3 一致：
  ``infovqa_val_lite``, ``mmmu_val``, ``ocrbench``）。

**环境变量（建议）**

.. code-block:: bash

    export HF_HOME=~/fsas/models/huggingface
    export HF_DATASETS_CACHE=~/fsas/datasets/deepseek-vl2-bridge/hf_datasets_cache
    export LMMS_EVAL_DATASETS_CACHE="$HF_DATASETS_CACHE"
    export HF_TOKEN="$(tr -d ' \\n\\r' < /path/to/hf_token.txt)"  # 去掉 token 文件中的空白与换行
    export PYTHONPATH=/root/DeepSeek-VL2:/root/DeepSeek-VL2/lmms-eval

**MMBench**：任务名如 ``mmbench_en_dev_lite`` 需在 ``--tasks`` 中显式加入；聚合若含
``gpt_eval_score`` 需配置 ``OPENAI_API_KEY`` 等。

**Score retention (%)**：对合并模型跑评测后，用 ``--baseline-json`` 指向基座
``lm_eval_results.json`` 或 ``lmms_eval_results.json``，会额外写出 ``retention.json``。
亦可用子命令 ``retention`` 仅做离线对比。

用法示例::

    # 文本基座
    python mystle/experiments/evaluate.py --backend text --run-id step5_base_lm

    # 多模态基座（全量去掉 --limit）
    python mystle/experiments/evaluate.py --backend mm --run-id step5_base_mm --limit 5

    # 离线 retention
    python mystle/experiments/evaluate.py retention \\
        --baseline ~/fsas/vlm/.../base/lmms_eval_results.json \\
        --current ~/fsas/vlm/.../merged/lmms_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_LMMS = _REPO_ROOT / "lmms-eval"
if _LMMS.is_dir() and str(_LMMS) not in sys.path:
    sys.path.insert(0, str(_LMMS))


def _default_results_dir() -> Path:
    return Path(os.environ.get("HOME", "/root")) / "fsas/vlm/deepseek-vl2-bridge/results"


def _default_text_tasks() -> str:
    return "arc_easy,winogrande,piqa"


def _default_mm_tasks() -> str:
    return "infovqa_val_lite,mmmu_val,ocrbench"


# 各任务优先用于 retention 的主指标键（lm-eval / lmms-eval 常见命名）
_PRIMARY_METRIC_KEYS: Tuple[str, ...] = (
    "acc,none",
    "acc_norm,none",
    "anls,none",
    "mmmu_acc,none",
    "ocrbench_accuracy,none",
    "gpt_eval_score,none",
)


def _is_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    return False


def find_primary_metric(task_result: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """从单任务结果 dict 中取一个主指标 (metric_key, value)。"""
    skip = {"alias", "name", "task", "samples", "sample_len", "version"}
    candidates: list[Tuple[str, float]] = []
    for k, v in task_result.items():
        if k in skip or k == " ":
            continue
        if "stderr" in k:
            continue
        if _is_number(v):
            candidates.append((k, float(v)))

    for pk in _PRIMARY_METRIC_KEYS:
        for k, val in candidates:
            if k == pk:
                return (k, val)
    # 前缀匹配（如变体）
    for pk in _PRIMARY_METRIC_KEYS:
        base = pk.split(",")[0]
        for k, val in candidates:
            if k.startswith(base + ",") or k == base:
                return (k, val)
    if candidates:
        return candidates[0]
    return None


def extract_task_scores(results_blob: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    从 lm-eval / lmms-eval 导出的精简 JSON（含 ``results`` 键）提取各任务主指标。
    返回 ``task_name -> {metric_key, value, alias}``。
    """
    inner = results_blob.get("results")
    if not isinstance(inner, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for task_name, task_result in inner.items():
        if not isinstance(task_result, dict):
            continue
        pm = find_primary_metric(task_result)
        if pm is None:
            continue
        mk, mv = pm
        out[task_name] = {
            "metric_key": mk,
            "value": mv,
            "alias": task_result.get("alias", task_name),
        }
    return out


def compute_retention_pct(
    baseline: Dict[str, Dict[str, Any]],
    current: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """按任务对齐计算 current 相对 baseline 的 retention（百分比）。"""
    rows = []
    for task, binfo in baseline.items():
        if task not in current:
            rows.append(
                {
                    "task": task,
                    "status": "missing_in_current",
                    "baseline": binfo.get("value"),
                    "current": None,
                    "retention_pct": None,
                }
            )
            continue
        cinfo = current[task]
        bv = float(binfo["value"])
        cv = float(cinfo["value"])
        if bv == 0:
            rp = None
            note = "baseline_zero"
        else:
            rp = 100.0 * cv / bv
            note = "ok"
        rows.append(
            {
                "task": task,
                "metric_key": binfo.get("metric_key"),
                "baseline": bv,
                "current": cv,
                "retention_pct": rp,
                "note": note,
            }
        )
    for task in current:
        if task not in baseline:
            rows.append(
                {
                    "task": task,
                    "status": "only_in_current",
                    "baseline": None,
                    "current": current[task].get("value"),
                    "retention_pct": None,
                }
            )
    return {"per_task": rows}


def _write_retention_json(
    path: Path,
    baseline_path: Path,
    current_path: Path,
    retention: Dict[str, Any],
) -> None:
    payload = {
        "baseline_json": str(baseline_path),
        "current_json": str(current_path),
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        **retention,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def cmd_retention(args: argparse.Namespace) -> None:
    with open(args.baseline, encoding="utf-8") as f:
        base_blob = json.load(f)
    with open(args.current, encoding="utf-8") as f:
        cur_blob = json.load(f)
    bscores = extract_task_scores(base_blob)
    cscores = extract_task_scores(cur_blob)
    ret = compute_retention_pct(bscores, cscores)
    out = Path(args.output) if args.output else Path(args.current).parent / "retention.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_retention_json(out, Path(args.baseline), Path(args.current), ret)
    print(f"Wrote {out}")


def _register_deepseek_vl2_mm() -> None:
    import mystle.experiments.deepseek_vl2_lmms  # noqa: F401

    from lmms_eval.models import MODEL_REGISTRY_V2
    from lmms_eval.models.registry_v2 import ModelManifest

    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(
            model_id="deepseek_vl2",
            simple_class_path="mystle.experiments.deepseek_vl2_lmms.DeepseekVL2LMMS",
        ),
        overwrite=True,
    )


def run_text_eval(args: argparse.Namespace) -> Path:
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
            json.dump(results, f, indent=2, ensure_ascii=False, default=handle_non_serializable)

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

    if args.baseline_json:
        with open(args.baseline_json, encoding="utf-8") as f:
            base_blob = json.load(f)
        ret = compute_retention_pct(extract_task_scores(base_blob), extract_task_scores(compact))
        _write_retention_json(
            out_dir / "retention.json",
            Path(args.baseline_json).resolve(),
            summary_path.resolve(),
            ret,
        )
        print(f"Retention vs baseline: {out_dir / 'retention.json'}")

    print(f"Done. Results: {summary_path}")
    return summary_path


def run_mm_eval(args: argparse.Namespace) -> Path:
    _register_deepseek_vl2_mm()
    from lmms_eval.evaluator import simple_evaluate
    from lmms_eval.utils import handle_non_serializable

    results_root = args.results_root or _default_results_dir()
    run_id = args.run_id or f"step5_mm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = results_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model_args = (
        f"pretrained={args.pretrained},"
        f"dtype={args.dtype},"
        f"chunk_size={args.chunk_size}"
    )
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
        device=args.device,
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

    if args.baseline_json:
        with open(args.baseline_json, encoding="utf-8") as f:
            base_blob = json.load(f)
        ret = compute_retention_pct(extract_task_scores(base_blob), extract_task_scores(compact))
        _write_retention_json(
            out_dir / "retention.json",
            Path(args.baseline_json).resolve(),
            summary_path.resolve(),
            ret,
        )
        print(f"Retention vs baseline: {out_dir / 'retention.json'}")

    print(f"Done. Results: {summary_path}")
    return summary_path


def _build_eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Step5：DeepSeek-VL2 测评（lm-eval 文本 / lmms-eval 多模态）"
    )
    p.add_argument(
        "--backend",
        choices=("text", "mm"),
        default="text",
        help="text=lm-eval；mm=lmms-eval 多模态",
    )
    p.add_argument("--pretrained", default="deepseek-ai/deepseek-vl2-small")
    p.add_argument(
        "--tasks",
        default=None,
        help="逗号分隔任务名；未指定时：text 默认 arc_easy,winogrande,piqa；mm 默认 infovqa_val_lite,mmmu_val,ocrbench",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="默认 text=4，mm=1（VL2 lmms 适配器按 batch=1 验证）",
    )
    p.add_argument(
        "--limit",
        type=float,
        default=None,
        help="每任务样本上限；lm-eval 为 int；lmms 可为 float，<1 表示比例",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-id", default=None)
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="基座结果 JSON，用于计算 retention.json（任务名需可对齐）",
    )
    p.add_argument(
        "--save-full",
        action="store_true",
        help="写入完整评测输出（含 configs[mm]，体积可能很大）",
    )
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help="lmms bootstrap 次数；0 不计算 stderr",
    )
    return p


def _build_retention_parser() -> argparse.ArgumentParser:
    rp = argparse.ArgumentParser(
        prog="evaluate.py retention",
        description="离线计算两组评测 JSON 的 score retention (%)",
    )
    rp.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="基座 lm_eval_results.json 或 lmms_eval_results.json",
    )
    rp.add_argument("--current", type=Path, required=True, help="当前模型结果 JSON")
    rp.add_argument(
        "--output",
        type=Path,
        default=None,
        help="默认写在 current 同目录 retention.json",
    )
    return rp


def _apply_eval_defaults(ns: argparse.Namespace) -> None:
    if ns.tasks is None:
        ns.tasks = _default_text_tasks() if ns.backend == "text" else _default_mm_tasks()
    if ns.batch_size is None:
        ns.batch_size = 4 if ns.backend == "text" else 1
    if ns.backend == "text" and ns.limit is not None:
        ns.limit = int(ns.limit)


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "retention":
        args = _build_retention_parser().parse_args(argv[1:])
        cmd_retention(args)
        return

    # 可选：第一个位置参数 text/mm 等价于 --backend
    if argv and argv[0] in ("text", "mm"):
        backend = argv[0]
        argv = argv[1:]
    else:
        backend = None

    args_ns = _build_eval_parser().parse_args(argv)
    if backend is not None:
        args_ns.backend = backend
    _apply_eval_defaults(args_ns)

    if args_ns.backend == "text":
        run_text_eval(args_ns)
    else:
        run_mm_eval(args_ns)


if __name__ == "__main__":
    main()
