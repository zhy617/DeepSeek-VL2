#!/usr/bin/env python3
"""
REAP-style expert pruning for DeepSeek-VL2 MoE.

Unlike merge-based methods (HC-SMoE, MC-SMoE) that cluster and average experts,
REAP removes the least important experts entirely. Expert importance is measured
via routing frequency (how often the gate selects each expert).

Supports optional --bridge flag to protect cross-modal bridge experts from
pruning (REAP+Bridge variant).

The pruning approach:
  1. Collect routing frequencies via gate probing
  2. Rank experts by importance (routing frequency)
  3. For Bridge variant: boost importance of high-bridge-score experts
  4. Keep top-N experts, remove the rest
  5. Rebuild MoE modules with only surviving experts

Reference:
  Lu et al. "Not All Experts are Equal: Efficient Expert Pruning and Skipping
  for Mixture of Experts Large Language Models" ACL 2024.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import moe_merge_core as mmc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="REAP: Routing-frequency Expert Activation Pruning for DeepSeek-VL2."
    )
    p.add_argument("--model-path", type=str, default="deepseek-ai/deepseek-vl2-small")
    p.add_argument(
        "--output-dir", type=str,
        default=os.path.expanduser("~/fsas/vlm/deepseek-vl2-bridge/results"),
    )
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--compression-keep", type=float, default=0.5,
        help="Fraction of experts to keep (0.5 = 50%%, 0.75 = 75%%)",
    )
    p.add_argument(
        "--probe-count", type=int, default=256,
        help="Number of probe hidden states for routing frequency estimation",
    )
    p.add_argument("--probe-batch-size", type=int, default=32)
    p.add_argument("--probe-scale", type=float, default=1.0)
    p.add_argument(
        "--bridge", action="store_true",
        help="Apply bridge-aware protection to prevent pruning cross-modal experts",
    )
    p.add_argument(
        "--layer-bridge-summary", type=str,
        default=os.path.expanduser(
            "~/fsas/vlm/deepseek-vl2-bridge/results/bridge_score_1000_full/layer_bridge_summary.json"
        ),
    )
    p.add_argument(
        "--bridge-protection-weight", type=float, default=1.0,
        help="Weight for bridge score in importance calculation (higher = more protection)",
    )
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--smoke-forward", action="store_true")
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


def collect_per_layer_routing_frequencies(
    moes: List[Any],
    probe_inputs: torch.Tensor,
    probe_batch_size: int,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      freq_matrix: [n_layers, E] - routing frequency per layer per expert
      global_freq: [E] - average routing frequency across all layers
    """
    n_experts = len(moes[0].experts)
    n_layers = len(moes)
    freq_matrix = np.zeros((n_layers, n_experts), dtype=np.float64)
    probe_batches = list(torch.split(probe_inputs, probe_batch_size, dim=0))
    total_tokens = 0

    with torch.no_grad():
        for layer_idx, moe in enumerate(moes):
            gate = moe.gate
            for pb in probe_batches:
                pb_3d = pb.unsqueeze(1)
                topk_idx, topk_weight, aux_loss = gate(pb_3d)
                selected = topk_idx.reshape(-1, topk_idx.shape[-1])

                for expert_idx in range(n_experts):
                    count = (selected == expert_idx).sum().item()
                    freq_matrix[layer_idx, expert_idx] += count
                if layer_idx == 0:
                    total_tokens += selected.shape[0]

    if total_tokens > 0:
        freq_matrix /= total_tokens

    global_freq = freq_matrix.mean(axis=0)
    return freq_matrix, global_freq


def compute_bridge_importance(
    layer_summary: Dict[str, Any],
    n_experts: int,
) -> np.ndarray:
    """
    Extract per-expert bridge importance from layer_bridge_summary.json.

    Uses B_z (bridge z-score) aggregated across layers: experts with higher
    mean |B_z| are more important for cross-modal processing.

    Returns: [E] normalized bridge importance scores (0-1 range).
    """
    layers = layer_summary.get("layers", {})
    if not layers:
        return np.zeros(n_experts)

    bz_accum = np.zeros(n_experts)
    n_counted = 0

    for layer_key, row in layers.items():
        bz = row.get("B_z", [])
        if len(bz) >= n_experts:
            bz_arr = np.array(bz[:n_experts], dtype=np.float64)
            bz_arr = np.where(np.isfinite(bz_arr), np.abs(bz_arr), 0.0)
            bz_accum += bz_arr
            n_counted += 1

    if n_counted > 0:
        bz_accum /= n_counted

    bz_max = bz_accum.max()
    if bz_max > 0:
        bz_accum /= bz_max

    return bz_accum


def compute_expert_importance(
    global_freq: np.ndarray,
    bridge_scores: Optional[np.ndarray] = None,
    bridge_weight: float = 1.0,
) -> np.ndarray:
    """
    Combine routing frequency with optional bridge protection.

    importance = freq_normalized + bridge_weight * bridge_normalized

    Higher importance = less likely to be pruned.
    """
    freq_norm = global_freq.copy()
    f_max = freq_norm.max()
    if f_max > 0:
        freq_norm /= f_max

    if bridge_scores is not None and bridge_weight > 0:
        importance = freq_norm + bridge_weight * bridge_scores
    else:
        importance = freq_norm

    return importance


def build_pruned_moe_module(
    old_moe: Any,
    keep_indices: List[int],
    lang_config: Any,
    device: torch.device,
) -> Any:
    """
    Build a new MoE module keeping only the specified experts.

    Unlike merging, pruning simply copies selected experts without averaging.
    """
    from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE

    new_moe = DeepseekV2MoE(lang_config).to(device)

    if old_moe.config.n_shared_experts is not None and new_moe.shared_experts is not None:
        new_moe.shared_experts.load_state_dict(old_moe.shared_experts.state_dict())

    for new_idx, old_idx in enumerate(keep_indices):
        sd = old_moe.experts[old_idx].state_dict()
        new_moe.experts[new_idx].load_state_dict(sd)

    with torch.no_grad():
        old_w = old_moe.gate.weight
        new_w = new_moe.gate.weight
        for new_idx, old_idx in enumerate(keep_indices):
            new_w[new_idx].copy_(old_w[old_idx])

        if (hasattr(old_moe.gate, "e_score_correction_bias") and
                hasattr(new_moe.gate, "e_score_correction_bias")):
            old_b = old_moe.gate.e_score_correction_bias
            new_b = new_moe.gate.e_score_correction_bias
            for new_idx, old_idx in enumerate(keep_indices):
                new_b[new_idx].copy_(old_b[old_idx])

    new_moe.eval()
    return new_moe


def apply_global_prune(
    language_model: nn.Module,
    keep_indices: List[int],
    device: torch.device,
) -> None:
    """Replace every MoE layer with a pruned version keeping only keep_indices."""
    from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE

    cfg = language_model.config
    cfg.n_routed_experts = len(keep_indices)

    for layer in language_model.model.layers:
        if not isinstance(layer.mlp, DeepseekV2MoE):
            continue
        old = layer.mlp
        layer.mlp = build_pruned_moe_module(old, keep_indices, cfg, device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_root = Path(args.output_dir).expanduser().resolve()
    keep_str = f"{args.compression_keep:.2f}".replace(".", "p").rstrip("0").rstrip("p") or "0"
    bridge_tag = "_bridge" if args.bridge else ""
    default_run_id = f"baselines/reap{bridge_tag}_keep_{keep_str}"
    run_id = args.run_id or default_run_id
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"[REAP] Loading model on CPU: {args.model_path}")

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    vl_gpt = vl_gpt.to(device).eval()
    lang = vl_gpt.language

    moes = mmc.iter_moe_modules(lang)
    if not moes:
        raise RuntimeError("No MoE layers found")

    cfg = lang.config
    n_old = int(cfg.n_routed_experts)
    n_group = getattr(cfg, "n_group", None)
    topk = int(cfg.num_experts_per_tok)
    target_n = mmc.resolve_target_n_routed(n_old, args.compression_keep, n_group, topk)

    print(f"[REAP] n_routed: {n_old} -> {target_n} (keep={args.compression_keep})")
    print(f"[REAP] Building probe hidden states (count={args.probe_count})")

    hidden_size = int(cfg.hidden_size)
    probe_inputs = build_probe_hidden_states(
        hidden_size, args.probe_count, args.seed, device, torch.bfloat16, args.probe_scale,
    )

    print(f"[REAP] Collecting routing frequencies across {len(moes)} MoE layers...")
    freq_matrix, global_freq = collect_per_layer_routing_frequencies(
        moes, probe_inputs, args.probe_batch_size, topk,
    )

    bridge_scores = None
    if args.bridge:
        print(f"[REAP+Bridge] Loading bridge summary: {args.layer_bridge_summary}")
        with open(args.layer_bridge_summary, "r") as f:
            layer_summary = json.load(f)
        bridge_scores = compute_bridge_importance(layer_summary, n_old)
        print(f"[REAP+Bridge] Bridge score range: [{bridge_scores.min():.4f}, {bridge_scores.max():.4f}]")

    importance = compute_expert_importance(
        global_freq, bridge_scores, args.bridge_protection_weight,
    )

    ranking = np.argsort(-importance)
    keep_indices = sorted(ranking[:target_n].tolist())
    pruned_indices = sorted(ranking[target_n:].tolist())

    print(f"[REAP] Keeping experts: {keep_indices}")
    print(f"[REAP] Pruning experts: {pruned_indices}")
    print(f"[REAP] Importance of kept: min={importance[keep_indices].min():.6f}, "
          f"max={importance[keep_indices].max():.6f}")
    print(f"[REAP] Importance of pruned: min={importance[pruned_indices].min():.6f}, "
          f"max={importance[pruned_indices].max():.6f}")

    meta: Dict[str, Any] = {
        "method": f"REAP{'+Bridge' if args.bridge else ''}",
        "compression_type": "pruning",
        "model_path": args.model_path,
        "n_routed_old": n_old,
        "n_routed_new": target_n,
        "compression_keep": args.compression_keep,
        "seed": args.seed,
        "probe_count": args.probe_count,
        "probe_batch_size": args.probe_batch_size,
        "probe_scale": args.probe_scale,
        "keep_indices": keep_indices,
        "pruned_indices": pruned_indices,
    }
    if args.bridge:
        meta.update({
            "bridge_summary_path": args.layer_bridge_summary,
            "bridge_protection_weight": args.bridge_protection_weight,
        })

    prune_plan = {
        "meta": meta,
        "n_keep": target_n,
        "keep_indices": keep_indices,
        "pruned_indices": pruned_indices,
        "importance_scores": importance.tolist(),
        "global_freq": global_freq.tolist(),
    }
    if bridge_scores is not None:
        prune_plan["bridge_scores"] = bridge_scores.tolist()

    with open(run_dir / "prune_plan.json", "w", encoding="utf-8") as f:
        json.dump(prune_plan, f, ensure_ascii=False, indent=2)
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    freq_info = {
        "global_freq": global_freq.tolist(),
        "freq_matrix_shape": list(freq_matrix.shape),
        "freq_matrix": freq_matrix.tolist(),
    }
    with open(run_dir / "routing_freq.json", "w") as f:
        json.dump(freq_info, f, indent=2)

    print(f"[REAP] Prune plan saved to {run_dir}")
    print(f"[REAP] Applying pruning...")

    apply_global_prune(lang, keep_indices, device)

    if args.smoke_forward:
        B, T, H = 1, 4, hidden_size
        dummy = torch.zeros(B, T, H, device=device, dtype=torch.bfloat16)
        mask = torch.ones(B, T, device=device, dtype=torch.long)
        with torch.no_grad():
            _ = lang.model(inputs_embeds=dummy, attention_mask=mask, use_cache=False)
        print("[REAP] Smoke forward passed.")

    if not args.no_save_model:
        save_path = run_dir / "merged_model"
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"[REAP] Saving pruned model to {save_path}...")
        vl_gpt.save_pretrained(str(save_path), safe_serialization=True)
        proc = DeepseekVLV2Processor.from_pretrained(args.model_path)
        proc.save_pretrained(str(save_path))

    print(f"[REAP] Done. n_routed {n_old} -> {target_n}. Output: {run_dir}")


if __name__ == "__main__":
    main()
