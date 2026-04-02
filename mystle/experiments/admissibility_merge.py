#!/usr/bin/env python3
"""
Step 4 — Layer-First + Expert-Refined Admissibility-Gated Merge（Ours）。

读取 bridge_score_results.json（可选）与 layer_bridge_summary.json，在高 σ_l 层上施加 B_z / M 可合并约束，
在约束后的距离矩阵上做层次聚类，再调用与 HC-SMoE 相同的权重平均合并。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import moe_merge_core as mmc


def resolve_tau_disp(s: str, layer_summary: Dict[str, Any]) -> float:
    """
    τ_disp：仅当 σ_l ≥ τ_disp 时在该层启用 per-expert B_z/M 约束。
    - `auto` / `auto_p50`：σ_l 的中位数
    - `auto_p75`（默认）：σ_l 的 75% 分位，仅约 top 25% 离散层施加约束（与「高 σ_l 层」一致）
    - `auto_p90`：更严，仅约 top 10% 层
    `0` 会使几乎所有层同时施加约束，跨层 AND 常导致无可合并对、聚类退化为单簇。
    """
    t = s.strip().lower()
    sigmas: list[float] = []
    for row in layer_summary.get("layers", {}).values():
        sl = row.get("sigma_l")
        if sl is not None and np.isfinite(sl):
            sigmas.append(float(sl))
    arr = np.array(sigmas, dtype=np.float64) if sigmas else np.array([], dtype=np.float64)

    if t in ("auto", "auto_p50"):
        if arr.size == 0:
            return 0.0
        return float(np.median(arr))
    if t == "auto_p75":
        if arr.size == 0:
            return 0.0
        return float(np.quantile(arr, 0.75))
    if t == "auto_p90":
        if arr.size == 0:
            return 0.0
        return float(np.quantile(arr, 0.90))
    return float(s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Admissibility-gated MoE merge (layer-first + expert-refined).")
    p.add_argument("--model-path", type=str, default="deepseek-ai/deepseek-vl2-small")
    p.add_argument(
        "--bridge-results",
        type=str,
        default="",
        help="bridge_score_results.json（可选，仅记录 meta）",
    )
    p.add_argument(
        "--layer-bridge-summary",
        type=str,
        default="",
        help="layer_bridge_summary.json（必填，含 B_z / M / sigma_l）",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/fsas/vlm/deepseek-vl2-bridge/results"),
    )
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compression-keep", type=float, default=0.5)
    p.add_argument(
        "--tau-disp",
        type=str,
        default="auto_p75",
        help="σ_l ≥ 该值时启用 per-expert B_z/M；默认 auto_p75=σ_l 的 75%% 分位；另有 auto/auto_p50/auto_p90 或显式浮点",
    )
    p.add_argument(
        "--tau-z",
        type=float,
        default=None,
        help="若设置：高 σ_l 层要求 B_z < 该值（绝对阈值）；与 --use-bz-quantile 互斥",
    )
    p.add_argument(
        "--use-bz-quantile",
        action="store_true",
        help="用每层 B_z 分位数作 cutoff（见 --bz-quantile-for-cutoff）",
    )
    p.add_argument(
        "--bz-quantile-for-cutoff",
        type=float,
        default=0.75,
        help="--use-bz-quantile 时：两专家 B_z 均 ≤ quantile(B_z, q) 才允许合并",
    )
    p.add_argument("--delta-affinity", type=float, default=2.0, help="|M_i - M_j| 上限")
    p.add_argument(
        "--forbidden-penalty",
        type=float,
        default=1e6,
        help="不可合并对的距离惩罚（代替无穷大以便 squareform）",
    )
    p.add_argument(
        "--admissibility-scope",
        type=str,
        default="max_sigma_layer",
        choices=("all", "max_sigma_layer"),
        help="B_z/M 约束作用范围：all=所有 σ_l≥τ_disp 的层（AND）；max_sigma_layer=仅 σ_l 最大的一层",
    )
    p.add_argument("--linkage", type=str, default="average", choices=("average", "complete", "single"))
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--smoke-forward", action="store_true")
    p.add_argument(
        "--merge-plan-only",
        action="store_true",
        help="仅聚类并写 merge_plan.json / meta.json，不将合并写入模型（省显存，与 hcsmoe_merge 一致）",
    )
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    summary_path = Path(args.layer_bridge_summary).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"请指定 --layer-bridge-summary 指向 layer_bridge_summary.json（当前: {summary_path}）"
        )
    layer_summary = load_json(summary_path)
    tau_disp = resolve_tau_disp(args.tau_disp, layer_summary)

    bridge_meta: Optional[Dict[str, Any]] = None
    if args.bridge_results:
        br = Path(args.bridge_results).expanduser().resolve()
        if br.is_file():
            bridge_meta = load_json(br)

    out_root = Path(args.output_dir).expanduser().resolve()
    run_id = args.run_id or time.strftime("admissibility_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    vl_gpt = vl_gpt.to(device).eval()
    lang = vl_gpt.language

    moes = mmc.iter_moe_modules(lang)
    if not moes:
        raise RuntimeError("未发现 MoE 层")

    cfg = lang.config
    n_old = int(cfg.n_routed_experts)
    n_group = getattr(cfg, "n_group", None)
    topk = int(cfg.num_experts_per_tok)

    target_n = mmc.resolve_target_n_routed(n_old, args.compression_keep, n_group, topk)

    X = mmc.aggregate_expert_vectors_across_moe_layers(moes)
    n_experts = X.shape[0]

    use_q = args.use_bz_quantile or args.tau_z is None
    tau_z_val = args.tau_z if args.tau_z is not None else 0.0

    D = mmc.constrained_distance_matrix(
        X,
        n_experts,
        layer_summary,
        tau_disp=tau_disp,
        tau_z=None if use_q else tau_z_val,
        delta_affinity=args.delta_affinity,
        use_per_layer_tau_z=use_q,
        bz_quantile_for_cutoff=args.bz_quantile_for_cutoff,
        forbidden_penalty=args.forbidden_penalty,
        admissibility_scope=args.admissibility_scope,
    )

    groups = mmc.hierarchical_cluster_from_full_distance(
        D, n_clusters=target_n, linkage_method=args.linkage
    )
    if len(groups) == 1 and target_n > 1 and n_experts > 1:
        warnings.warn(
            "约束距离下层次聚类退化为单簇（常见于 τ_disp 过小或禁并过多）；"
            "回退为与 HC-SMoE 相同的无约束余弦层次聚类。",
            UserWarning,
            stacklevel=1,
        )
        groups = mmc.hierarchical_cluster_groups(X, n_clusters=target_n, linkage_method=args.linkage)
        clustering_fallback = "unconstrained_cosine"
    else:
        clustering_fallback = None

    meta: Dict[str, Any] = {
        "method": "admissibility-gated-merge",
        "model_path": args.model_path,
        "layer_bridge_summary": str(summary_path),
        "n_routed_old": n_old,
        "n_routed_new": len(groups),
        "compression_keep": args.compression_keep,
        "tau_disp_arg": args.tau_disp,
        "tau_disp": tau_disp,
        "admissibility_scope": args.admissibility_scope,
        "tau_z": args.tau_z,
        "use_bz_quantile": use_q,
        "bz_quantile_for_cutoff": args.bz_quantile_for_cutoff,
        "delta_affinity": args.delta_affinity,
        "linkage": args.linkage,
        "seed": args.seed,
    }
    if bridge_meta is not None:
        meta["bridge_score_meta_keys"] = list(bridge_meta.keys())
    if clustering_fallback:
        meta["clustering_fallback"] = clustering_fallback

    mmc.save_merge_plan(run_dir / "merge_plan.json", groups, meta)

    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.merge_plan_only:
        del vl_gpt
        torch.cuda.empty_cache()
        print(f"merge-plan-only: n_routed {n_old} -> {len(groups)}. Output: {run_dir}")
        return

    mmc.apply_global_merge_partition(lang, groups, device)

    if args.smoke_forward:
        B, T, H = 1, 4, cfg.hidden_size
        dummy = torch.zeros(B, T, H, device=device, dtype=torch.bfloat16)
        mask = torch.ones(B, T, device=device, dtype=torch.long)
        with torch.no_grad():
            _ = lang.model(inputs_embeds=dummy, attention_mask=mask, use_cache=False)

    if not args.no_save_model:
        save_path = run_dir / "merged_model"
        save_path.mkdir(parents=True, exist_ok=True)
        vl_gpt.save_pretrained(str(save_path), safe_serialization=True)
        proc = DeepseekVLV2Processor.from_pretrained(args.model_path)
        proc.save_pretrained(str(save_path))

    print(f"Done. n_routed {n_old} -> {len(groups)}. Output: {run_dir}")


if __name__ == "__main__":
    main()
