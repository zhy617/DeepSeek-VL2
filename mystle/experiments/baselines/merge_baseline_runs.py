#!/usr/bin/env python3
"""
将同一方法（如 HC-SMoE）下多个压缩率 run 目录合并为一份 JSON，便于对比与 admissibility 引用。

用法:
  python merge_baseline_runs.py --output ~/fsas/vlm/deepseek-vl2-bridge/results/baselines/hcsmoe_all_compression.json \\
    ~/fsas/.../baselines/hcsmoe_keep_0p25 ~/fsas/.../baselines/hcsmoe_keep_0p50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge HC-SMoE (or similar) baseline run folders into one JSON.")
    p.add_argument(
        "run_dirs",
        nargs="+",
        type=str,
        help="各压缩率对应的输出目录（含 meta.json 与 merge_plan.json）",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="合并后的 JSON 路径（建议放在 fsas .../results/baselines/）",
    )
    p.add_argument(
        "--method-label",
        type=str,
        default="HC-SMoE",
        help="写入汇总的 method 字段",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    entries: List[Dict[str, Any]] = []
    for d in args.run_dirs:
        rd = Path(d).expanduser().resolve()
        meta_path = rd / "meta.json"
        plan_path = rd / "merge_plan.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"缺少 meta.json: {meta_path}")
        if not plan_path.is_file():
            raise FileNotFoundError(f"缺少 merge_plan.json: {plan_path}")
        meta = _load_json(meta_path)
        plan = _load_json(plan_path)
        ck = meta.get("compression_keep")
        entries.append(
            {
                "run_dir": str(rd),
                "compression_keep": ck,
                "meta": meta,
                "merge_plan": plan,
            }
        )

    entries.sort(key=lambda x: (x.get("compression_keep") is None, x.get("compression_keep", 0)))

    merged: Dict[str, Any] = {
        "method": args.method_label,
        "runs": entries,
        "note": "各 compression_keep 一次 run；merge_plan 为完整划分，供复现或下游加载",
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out} ({len(entries)} runs)")


if __name__ == "__main__":
    main()
