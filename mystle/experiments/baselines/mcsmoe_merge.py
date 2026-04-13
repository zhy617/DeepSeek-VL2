#!/usr/bin/env python3
"""
MC-SMoE (ICLR 2024 Spotlight) expert merging for DeepSeek-VL2.

Key difference from HC-SMoE:
  - Feature: routing-frequency vectors (how often each expert is selected per layer)
  - Merge: frequency-weighted average instead of simple average

Supports optional --bridge flag to apply bridge admissibility constraints
(MC-SMoE+Bridge variant) for multimodal-aware compression.

Reference:
  Li et al. "Merge, Then Compress: Demystifying Optimal Data-Free Expert Merging
  for Mixture-of-Experts Language Models" ICLR 2024 Spotlight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import moe_merge_core as mmc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MC-SMoE: routing-frequency-weighted expert merging."
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
        "--linkage", type=str, default="average",
        choices=("average", "complete", "single"),
    )
    p.add_argument(
        "--probe-count", type=int, default=256,
        help="Number of probe hidden states for routing frequency estimation",
    )
    p.add_argument(
        "--probe-batch-size", type=int, default=32,
        help="Batch size for probing the gate",
    )
    p.add_argument(
        "--probe-scale", type=float, default=1.0,
        help="Scale for Gaussian probes (actual scale = probe_scale / sqrt(hidden_size))",
    )
    p.add_argument(
        "--bridge", action="store_true",
        help="Apply bridge admissibility constraints (MC-SMoE+Bridge variant)",
    )
    p.add_argument(
        "--layer-bridge-summary", type=str,
        default=os.path.expanduser(
            "~/fsas/vlm/deepseek-vl2-bridge/results/bridge_score_1000_full/layer_bridge_summary.json"
        ),
        help="Path to layer_bridge_summary.json (only used with --bridge)",
    )
    p.add_argument("--tau-disp", type=str, default="auto_p75")
    p.add_argument("--use-bz-quantile", action="store_true", default=True)
    p.add_argument("--bz-quantile-for-cutoff", type=float, default=0.75)
    p.add_argument("--delta-affinity", type=float, default=2.0)
    p.add_argument(
        "--admissibility-scope", type=str, default="max_sigma_layer",
        choices=("all", "max_sigma_layer"),
    )
    p.add_argument(
        "--forbidden-penalty", type=float, default=10.0,
        help="Penalty for inadmissible pairs in the distance matrix",
    )
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--smoke-forward", action="store_true")
    p.add_argument("--merge-plan-only", action="store_true")
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


def collect_routing_frequencies(
    moes: List[Any],
    probe_inputs: torch.Tensor,
    probe_batch_size: int,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect routing frequencies by passing probes through each MoE gate.

    The gate expects 3D input [bsz, seq_len, hidden_size] and returns
    (topk_idx, topk_weight, aux_loss).

    Returns:
      freq_matrix: [E, n_layers] - routing frequency per expert per layer
      total_freq:  [E]           - total routing frequency across all layers
    """
    n_experts = len(moes[0].experts)
    n_layers = len(moes)
    freq_matrix = np.zeros((n_experts, n_layers), dtype=np.float64)
    probe_batches = list(torch.split(probe_inputs, probe_batch_size, dim=0))
    total_tokens = 0

    with torch.no_grad():
        for layer_idx, moe in enumerate(moes):
            gate = moe.gate
            for pb in probe_batches:
                pb_3d = pb.unsqueeze(1)  # [batch, 1, hidden] — single-token sequences
                topk_idx, topk_weight, aux_loss = gate(pb_3d)
                selected = topk_idx.reshape(-1, topk_idx.shape[-1])  # [batch, topk]

                for expert_idx in range(n_experts):
                    count = (selected == expert_idx).sum().item()
                    freq_matrix[expert_idx, layer_idx] += count
                if layer_idx == 0:
                    total_tokens += selected.shape[0]

    if total_tokens > 0:
        freq_matrix /= total_tokens

    total_freq = freq_matrix.sum(axis=1)
    norm = total_freq.sum()
    if norm > 0:
        total_freq /= norm

    return freq_matrix, total_freq


def frequency_weighted_average_mlp(
    mlps: Sequence[Any],
    weights: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Frequency-weighted average of expert MLP parameters."""
    ref = mlps[0].state_dict()
    w = weights / weights.sum() if weights.sum() > 0 else np.ones(len(mlps)) / len(mlps)
    out: Dict[str, torch.Tensor] = {}
    for k in ref.keys():
        stacked = torch.stack([m.state_dict()[k].float() for m in mlps], dim=0)
        w_tensor = torch.tensor(w, dtype=torch.float32, device=stacked.device).reshape(-1, *([1] * (stacked.dim() - 1)))
        out[k] = (stacked * w_tensor).sum(0).to(ref[k].dtype)
    return out


def frequency_weighted_gate_merge(
    old_gate: Any,
    groups: Sequence[Sequence[int]],
    new_gate: Any,
    total_freq: np.ndarray,
) -> None:
    """Merge gate weights using frequency-weighted average."""
    with torch.no_grad():
        ow = old_gate.weight
        nw = new_gate.weight
        for ni, g in enumerate(groups):
            idx = list(g)
            w = total_freq[idx]
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones(len(idx)) / len(idx)
            w_t = torch.tensor(w, dtype=torch.float32, device=ow.device).reshape(-1, 1)
            nw[ni].copy_((ow[idx].float() * w_t).sum(0).to(nw.dtype))

        if hasattr(old_gate, "e_score_correction_bias") and hasattr(new_gate, "e_score_correction_bias"):
            ob = old_gate.e_score_correction_bias
            nb = new_gate.e_score_correction_bias
            for ni, g in enumerate(groups):
                idx = list(g)
                w = total_freq[idx]
                w_sum = w.sum()
                if w_sum > 0:
                    w = w / w_sum
                else:
                    w = np.ones(len(idx)) / len(idx)
                w_t = torch.tensor(w, dtype=torch.float32, device=ob.device)
                nb[ni].copy_((ob[idx].float() * w_t).sum().to(nb.dtype))


def build_freq_weighted_merged_moe(
    old_moe: Any,
    groups: List[List[int]],
    lang_config: Any,
    device: torch.device,
    total_freq: np.ndarray,
) -> Any:
    """Build a new MoE module with frequency-weighted merged experts."""
    from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE

    new_moe = DeepseekV2MoE(lang_config).to(device)
    if old_moe.config.n_shared_experts is not None and new_moe.shared_experts is not None:
        new_moe.shared_experts.load_state_dict(old_moe.shared_experts.state_dict())

    for ni, g in enumerate(groups):
        mlps = [old_moe.experts[i] for i in g]
        w = total_freq[list(g)]
        avg_sd = frequency_weighted_average_mlp(mlps, w)
        new_moe.experts[ni].load_state_dict(avg_sd)

    frequency_weighted_gate_merge(old_moe.gate, groups, new_moe.gate, total_freq)
    new_moe.eval()
    return new_moe


def apply_freq_weighted_merge_partition(
    language_model: Any,
    groups: List[List[int]],
    device: torch.device,
    total_freq: np.ndarray,
) -> None:
    """Apply frequency-weighted merge to all MoE layers."""
    from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE

    cfg = language_model.config
    cfg.n_routed_experts = len(groups)

    for layer in language_model.model.layers:
        if not isinstance(layer.mlp, DeepseekV2MoE):
            continue
        old = layer.mlp
        layer.mlp = build_freq_weighted_merged_moe(old, groups, cfg, device, total_freq)


def resolve_tau_disp(tau_disp_str: str, layer_summary: Dict[str, Any]) -> float:
    """Resolve tau_disp from string (auto_pXX or float)."""
    if tau_disp_str.startswith("auto"):
        layers = layer_summary.get("layers", {})
        sigmas = [row["sigma_l"] for row in layers.values()
                  if "sigma_l" in row and np.isfinite(row["sigma_l"])]
        if not sigmas:
            return 0.0
        if "_p" in tau_disp_str:
            q = int(tau_disp_str.split("_p")[1]) / 100.0
        else:
            q = 0.5
        return float(np.quantile(sigmas, q))
    return float(tau_disp_str)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_root = Path(args.output_dir).expanduser().resolve()
    keep_str = f"{args.compression_keep:.2f}".replace(".", "p").rstrip("0").rstrip("p") or "0"
    bridge_tag = "_bridge" if args.bridge else ""
    default_run_id = f"baselines/mcsmoe{bridge_tag}_keep_{keep_str}"
    run_id = args.run_id or default_run_id
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MC-SMoE] Loading model: {args.model_path}")

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

    print(f"[MC-SMoE] n_routed: {n_old} -> {target_n} (keep={args.compression_keep})")
    print(f"[MC-SMoE] Building probe hidden states (count={args.probe_count})")

    hidden_size = int(cfg.hidden_size)
    probe_inputs = build_probe_hidden_states(
        hidden_size, args.probe_count, args.seed, device, torch.bfloat16, args.probe_scale,
    )

    print(f"[MC-SMoE] Collecting routing frequencies across {len(moes)} MoE layers...")
    freq_matrix, total_freq = collect_routing_frequencies(
        moes, probe_inputs, args.probe_batch_size, topk,
    )

    print(f"[MC-SMoE] Routing freq range: min={total_freq.min():.6f}, max={total_freq.max():.6f}")

    if args.bridge:
        print(f"[MC-SMoE+Bridge] Loading bridge summary: {args.layer_bridge_summary}")
        with open(args.layer_bridge_summary, "r") as f:
            layer_summary = json.load(f)

        tau_disp = resolve_tau_disp(args.tau_disp, layer_summary)
        print(f"[MC-SMoE+Bridge] tau_disp={tau_disp}, scope={args.admissibility_scope}")

        D = mmc.constrained_distance_matrix(
            freq_matrix, n_old, layer_summary,
            tau_disp=tau_disp,
            tau_z=None if args.use_bz_quantile else 0.0,
            delta_affinity=args.delta_affinity,
            use_per_layer_tau_z=args.use_bz_quantile,
            bz_quantile_for_cutoff=args.bz_quantile_for_cutoff,
            forbidden_penalty=args.forbidden_penalty,
            admissibility_scope=args.admissibility_scope,
        )
        groups = mmc.hierarchical_cluster_from_full_distance(D, target_n, args.linkage)
        constraint_method = "bridge_admissibility"

        n_forbidden = 0
        for i in range(n_old):
            for j in range(i + 1, n_old):
                if D[i, j] >= args.forbidden_penalty * 0.9:
                    n_forbidden += 1
        print(f"[MC-SMoE+Bridge] Forbidden pairs: {n_forbidden}/{n_old*(n_old-1)//2}")
    else:
        groups = mmc.hierarchical_cluster_groups(
            freq_matrix, n_clusters=target_n, linkage_method=args.linkage,
        )
        constraint_method = "none"

    meta: Dict[str, Any] = {
        "method": f"MC-SMoE{'+Bridge' if args.bridge else ''}",
        "model_path": args.model_path,
        "n_routed_old": n_old,
        "n_routed_new": len(groups),
        "compression_keep": args.compression_keep,
        "linkage": args.linkage,
        "seed": args.seed,
        "probe_count": args.probe_count,
        "probe_batch_size": args.probe_batch_size,
        "probe_scale": args.probe_scale,
        "merge_weighting": "frequency_weighted",
        "constraint_method": constraint_method,
    }
    if args.bridge:
        meta.update({
            "bridge_summary_path": args.layer_bridge_summary,
            "tau_disp": str(args.tau_disp),
            "tau_disp_resolved": tau_disp,
            "admissibility_scope": args.admissibility_scope,
            "bz_quantile_for_cutoff": args.bz_quantile_for_cutoff,
            "delta_affinity": args.delta_affinity,
            "forbidden_penalty": args.forbidden_penalty,
        })

    mmc.save_merge_plan(run_dir / "merge_plan.json", groups, meta)
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    freq_info = {
        "total_freq": total_freq.tolist(),
        "freq_matrix_shape": list(freq_matrix.shape),
        "freq_matrix": freq_matrix.tolist(),
    }
    with open(run_dir / "routing_freq.json", "w") as f:
        json.dump(freq_info, f, indent=2)

    print(f"[MC-SMoE] Merge plan saved: {len(groups)} groups")

    if args.merge_plan_only:
        del vl_gpt
        torch.cuda.empty_cache()
        print(f"[MC-SMoE] merge-plan-only done. Output: {run_dir}")
        return

    print(f"[MC-SMoE] Applying frequency-weighted merge...")
    apply_freq_weighted_merge_partition(lang, groups, device, total_freq)

    if args.smoke_forward:
        B, T, H = 1, 4, hidden_size
        dummy = torch.zeros(B, T, H, device=device, dtype=torch.bfloat16)
        mask = torch.ones(B, T, device=device, dtype=torch.long)
        with torch.no_grad():
            _ = lang.model(inputs_embeds=dummy, attention_mask=mask, use_cache=False)
        print("[MC-SMoE] Smoke forward passed.")

    if not args.no_save_model:
        save_path = run_dir / "merged_model"
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"[MC-SMoE] Saving merged model to {save_path}...")
        vl_gpt.save_pretrained(str(save_path), safe_serialization=True)
        proc = DeepseekVLV2Processor.from_pretrained(args.model_path)
        proc.save_pretrained(str(save_path))

    print(f"[MC-SMoE] Done. n_routed {n_old} -> {len(groups)}. Output: {run_dir}")


if __name__ == "__main__":
    main()
