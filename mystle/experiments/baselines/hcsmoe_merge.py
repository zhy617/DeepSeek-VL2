#!/usr/bin/env python3
"""
Step 4 — HC-SMoE baseline：按 expert 权重（跨 MoE 层平均）余弦距离做层次聚类，加权平均合并 MLP 与 Gate。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import moe_merge_core as mmc

aggregate_expert_vectors_across_moe_layers = mmc.aggregate_expert_vectors_across_moe_layers
apply_global_merge_partition = mmc.apply_global_merge_partition
hierarchical_cluster_groups = mmc.hierarchical_cluster_groups
iter_moe_modules = mmc.iter_moe_modules
resolve_target_n_routed = mmc.resolve_target_n_routed
save_merge_plan = mmc.save_merge_plan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HC-SMoE: hierarchical clustering merge for DeepSeek-VL2 MoE.")
    p.add_argument("--model-path", type=str, default="deepseek-ai/deepseek-vl2-small")
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/fsas/vlm/deepseek-vl2-bridge/results"),
    )
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--compression-keep",
        type=float,
        default=0.5,
        help="保留专家比例：0.5≈50%% 专家数；0.25≈25%%",
    )
    p.add_argument("--linkage", type=str, default="average", choices=("average", "complete", "single"))
    p.add_argument("--no-save-model", action="store_true", help="只写 merge_plan.json，不写完整权重")
    p.add_argument("--smoke-forward", action="store_true", help="合并后对空序列做一次 language.model forward")
    p.add_argument(
        "--merge-plan-only",
        action="store_true",
        help="仅聚类并写 merge_plan.json / meta.json，不将合并写入模型（省显存，适合多压缩率批量出方案）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_root = Path(args.output_dir).expanduser().resolve()
    run_id = args.run_id or time.strftime("hcsmoe_%Y%m%d_%H%M%S")
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

    moes = iter_moe_modules(lang)
    if not moes:
        raise RuntimeError("未发现 MoE 层")

    cfg = lang.config
    n_old = int(cfg.n_routed_experts)
    n_group = getattr(cfg, "n_group", None)
    topk = int(cfg.num_experts_per_tok)

    target_n = resolve_target_n_routed(n_old, args.compression_keep, n_group, topk)
    X = aggregate_expert_vectors_across_moe_layers(moes)

    groups = hierarchical_cluster_groups(X, n_clusters=target_n, linkage_method=args.linkage)

    meta: Dict[str, Any] = {
        "method": "HC-SMoE",
        "model_path": args.model_path,
        "n_routed_old": n_old,
        "n_routed_new": len(groups),
        "compression_keep": args.compression_keep,
        "linkage": args.linkage,
        "seed": args.seed,
    }
    save_merge_plan(run_dir / "merge_plan.json", groups, meta)

    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.merge_plan_only:
        del vl_gpt
        torch.cuda.empty_cache()
        print(f"merge-plan-only: n_routed {n_old} -> {len(groups)}. Output: {run_dir}")
        return

    apply_global_merge_partition(lang, groups, device)

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
