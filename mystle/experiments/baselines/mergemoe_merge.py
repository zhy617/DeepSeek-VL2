#!/usr/bin/env python3
"""
Step 4 / Step 7 — MergeMoE baseline。

与 HC-SMoE 共享同一权重平均合并与保存逻辑，但聚类特征改为 expert 的
"输出空间签名"：对固定 probe hidden states 经过各 expert MLP 的输出做
跨层平均，再按余弦距离做层次聚类。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import moe_merge_core as mmc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MergeMoE baseline: cluster experts by output-space signatures."
    )
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
    p.add_argument(
        "--probe-count",
        type=int,
        default=32,
        help="用于构造输出空间签名的 probe hidden states 数量",
    )
    p.add_argument(
        "--probe-batch-size",
        type=int,
        default=8,
        help="分批送入 expert 的 probe batch，降低额外显存",
    )
    p.add_argument(
        "--probe-scale",
        type=float,
        default=1.0,
        help="高斯 probe 的标准差缩放；实际会再除以 sqrt(hidden_size)",
    )
    p.add_argument("--no-save-model", action="store_true", help="只写 merge_plan.json，不写完整权重")
    p.add_argument("--smoke-forward", action="store_true", help="合并后对空序列做一次 language.model forward")
    p.add_argument(
        "--merge-plan-only",
        action="store_true",
        help="仅聚类并写 merge_plan.json / meta.json，不将合并写入模型（省显存）",
    )
    return p.parse_args()


def build_probe_hidden_states(
    hidden_size: int,
    probe_count: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    scale: float,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(probe_count, hidden_size, generator=gen, dtype=torch.float32)
    x = x * (scale / math.sqrt(float(hidden_size)))
    return x.to(device=device, dtype=dtype)


def aggregate_expert_output_signatures_across_moe_layers(
    moes: List[Any],
    probe_inputs: torch.Tensor,
    probe_batch_size: int,
) -> np.ndarray:
    """返回形状 [E, probe_count * hidden_size] 的 expert 输出签名。"""
    if not moes:
        raise ValueError("empty moes")

    n_experts = len(moes[0].experts)
    signatures: List[np.ndarray] = []
    probe_batches = list(torch.split(probe_inputs, probe_batch_size, dim=0))

    with torch.no_grad():
        for expert_idx in range(n_experts):
            layer_acc: List[torch.Tensor] = []
            for moe in moes:
                per_batch: List[torch.Tensor] = []
                expert = moe.experts[expert_idx]
                for pb in probe_batches:
                    out = expert(pb)
                    per_batch.append(out.float().cpu())
                layer_acc.append(torch.cat(per_batch, dim=0))
            mean_out = torch.stack(layer_acc, dim=0).mean(dim=0)
            signatures.append(mean_out.reshape(-1).numpy())
    return np.stack(signatures, axis=0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_root = Path(args.output_dir).expanduser().resolve()
    run_id = args.run_id or time.strftime("mergemoe_%Y%m%d_%H%M%S")
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

    probe_inputs = build_probe_hidden_states(
        hidden_size=int(cfg.hidden_size),
        probe_count=args.probe_count,
        seed=args.seed,
        device=device,
        dtype=torch.bfloat16,
        scale=args.probe_scale,
    )
    X = aggregate_expert_output_signatures_across_moe_layers(
        moes=moes,
        probe_inputs=probe_inputs,
        probe_batch_size=max(1, args.probe_batch_size),
    )
    groups = mmc.hierarchical_cluster_groups(X, n_clusters=target_n, linkage_method=args.linkage)

    meta: Dict[str, Any] = {
        "method": "MergeMoE",
        "model_path": args.model_path,
        "n_routed_old": n_old,
        "n_routed_new": len(groups),
        "compression_keep": args.compression_keep,
        "linkage": args.linkage,
        "seed": args.seed,
        "probe_count": args.probe_count,
        "probe_batch_size": args.probe_batch_size,
        "probe_scale": args.probe_scale,
        "signature_type": "expert_output_space",
    }
    mmc.save_merge_plan(run_dir / "merge_plan.json", groups, meta)

    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.merge_plan_only:
        del probe_inputs
        del vl_gpt
        torch.cuda.empty_cache()
        print(f"merge-plan-only: n_routed {n_old} -> {len(groups)}. Output: {run_dir}")
        return

    mmc.apply_global_merge_partition(lang, groups, device)

    if args.smoke_forward:
        batch_size, seq_len, hidden = 1, 4, int(cfg.hidden_size)
        dummy = torch.zeros(batch_size, seq_len, hidden, device=device, dtype=torch.bfloat16)
        mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
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
