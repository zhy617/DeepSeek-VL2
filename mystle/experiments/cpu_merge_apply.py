#!/usr/bin/env python3
"""
CPU-based merge applier for DeepSeek-VL2 MoE models.

Solves the GPU OOM problem by:
  1. Loading model on CPU (bf16)
  2. Generating merge plan (clustering) — lightweight, no GPU needed
  3. Applying merge partition on CPU
  4. Saving the merged model from CPU

This replaces the GPU-based apply_global_merge_partition when the full model
doesn't fit in VRAM alongside the new MoE modules.
"""

from __future__ import annotations

import argparse
import gc
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

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU-based merge for DeepSeek-VL2 MoE. Avoids GPU OOM."
    )
    p.add_argument("--model-path", type=str, default="deepseek-ai/deepseek-vl2-small")
    p.add_argument(
        "--output-dir", type=str,
        default=os.path.expanduser("~/fsas/vlm/deepseek-vl2-bridge/results"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--method", type=str, required=True,
        choices=("hcsmoe", "mergemoe", "admissibility", "mcsmoe", "mcsmoe_bridge"),
        help="Merge method to use",
    )
    p.add_argument(
        "--compression-keep", type=float, default=0.5,
        help="Fraction of experts to keep",
    )
    p.add_argument("--linkage", type=str, default="average")
    # MergeMoE / MC-SMoE probe params
    p.add_argument("--probe-count", type=int, default=256)
    p.add_argument("--probe-batch-size", type=int, default=32)
    p.add_argument("--probe-scale", type=float, default=1.0)
    # Admissibility / Bridge params
    p.add_argument(
        "--layer-bridge-summary", type=str,
        default=os.path.expanduser(
            "~/fsas/vlm/deepseek-vl2-bridge/results/bridge_score_1000_full/layer_bridge_summary.json"
        ),
    )
    p.add_argument("--tau-disp", type=str, default="auto_p75")
    p.add_argument("--use-bz-quantile", action="store_true", default=True)
    p.add_argument("--bz-quantile-for-cutoff", type=float, default=0.75)
    p.add_argument("--delta-affinity", type=float, default=2.0)
    p.add_argument("--admissibility-scope", type=str, default="max_sigma_layer")
    p.add_argument("--forbidden-penalty", type=float, default=10.0)
    return p.parse_args()


def resolve_tau_disp(tau_disp_str: str, layer_summary: Dict[str, Any]) -> float:
    if tau_disp_str.startswith("auto"):
        layers = layer_summary.get("layers", {})
        sigmas = [row["sigma_l"] for row in layers.values()
                  if "sigma_l" in row and np.isfinite(row["sigma_l"])]
        if not sigmas:
            return 0.0
        q = int(tau_disp_str.split("_p")[1]) / 100.0 if "_p" in tau_disp_str else 0.5
        return float(np.quantile(sigmas, q))
    return float(tau_disp_str)


def run_id_for(method: str, keep: float) -> str:
    keep_str = f"{keep:.2f}".replace(".", "p").rstrip("0").rstrip("p") or "0"
    method_tag = {
        "hcsmoe": "hcsmoe",
        "mergemoe": "mergemoe",
        "admissibility": "admissibility",
        "mcsmoe": "mcsmoe",
        "mcsmoe_bridge": "mcsmoe_bridge",
    }[method]
    return f"baselines/{method_tag}_keep_{keep_str}"


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    out_root = Path(args.output_dir).expanduser().resolve()
    run_id = run_id_for(args.method, args.compression_keep)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CPU-Merge] method={args.method}, keep={args.compression_keep}")
    print(f"[CPU-Merge] Output: {run_dir}")

    # --- Step 1: Load model on CPU in bf16 ---
    print("[CPU-Merge] Loading model on CPU (bf16)...")
    from transformers import AutoModelForCausalLM
    from deepseek_vl2.models import DeepseekVLV2Processor

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    vl_gpt = vl_gpt.eval()
    lang = vl_gpt.language
    print(f"[CPU-Merge] Model loaded on CPU")

    import moe_merge_core as mmc
    moes = mmc.iter_moe_modules(lang)
    if not moes:
        raise RuntimeError("No MoE layers found")

    cfg = lang.config
    n_old = int(cfg.n_routed_experts)
    n_group = getattr(cfg, "n_group", None)
    topk = int(cfg.num_experts_per_tok)
    target_n = mmc.resolve_target_n_routed(n_old, args.compression_keep, n_group, topk)
    hidden_size = int(cfg.hidden_size)

    print(f"[CPU-Merge] n_routed: {n_old} -> {target_n}")

    # --- Step 2: Compute features & clustering ---
    meta: Dict[str, Any] = {
        "model_path": args.model_path,
        "n_routed_old": n_old,
        "n_routed_new": target_n,
        "compression_keep": args.compression_keep,
        "linkage": args.linkage,
        "seed": args.seed,
        "cpu_merge": True,
    }

    if args.method == "hcsmoe":
        print("[CPU-Merge] HC-SMoE: weight-space clustering")
        X = mmc.aggregate_expert_vectors_across_moe_layers(moes)
        groups = mmc.hierarchical_cluster_groups(X, n_clusters=target_n, linkage_method=args.linkage)
        meta["method"] = "HC-SMoE"
        merge_weights = None

    elif args.method == "mergemoe":
        print("[CPU-Merge] MergeMoE: output-space clustering")
        probe_inputs = _build_probes(hidden_size, args.probe_count, args.seed, device, torch.bfloat16, args.probe_scale)
        X = _output_signatures(moes, probe_inputs, args.probe_batch_size)
        groups = mmc.hierarchical_cluster_groups(X, n_clusters=target_n, linkage_method=args.linkage)
        meta["method"] = "MergeMoE"
        meta["probe_count"] = args.probe_count
        merge_weights = None

    elif args.method == "admissibility":
        print("[CPU-Merge] Admissibility: bridge-constrained clustering")
        with open(args.layer_bridge_summary, "r") as f:
            layer_summary = json.load(f)
        tau_disp = resolve_tau_disp(args.tau_disp, layer_summary)
        X = mmc.aggregate_expert_vectors_across_moe_layers(moes)
        D = mmc.constrained_distance_matrix(
            X, n_old, layer_summary,
            tau_disp=tau_disp,
            tau_z=None if args.use_bz_quantile else 0.0,
            delta_affinity=args.delta_affinity,
            use_per_layer_tau_z=args.use_bz_quantile,
            bz_quantile_for_cutoff=args.bz_quantile_for_cutoff,
            forbidden_penalty=args.forbidden_penalty,
            admissibility_scope=args.admissibility_scope,
        )
        groups = mmc.hierarchical_cluster_from_full_distance(D, target_n, args.linkage)
        meta["method"] = "HC-SMoE+Bridge (admissibility)"
        meta["tau_disp"] = str(args.tau_disp)
        meta["tau_disp_resolved"] = tau_disp
        meta["admissibility_scope"] = args.admissibility_scope
        merge_weights = None

    elif args.method in ("mcsmoe", "mcsmoe_bridge"):
        is_bridge = args.method == "mcsmoe_bridge"
        tag = "MC-SMoE+Bridge" if is_bridge else "MC-SMoE"
        print(f"[CPU-Merge] {tag}: routing-frequency clustering")
        probe_inputs = _build_probes(hidden_size, args.probe_count, args.seed, device, torch.bfloat16, args.probe_scale)
        freq_matrix, total_freq = _collect_routing_freq(moes, probe_inputs, args.probe_batch_size, topk)
        merge_weights = total_freq

        freq_info = {"total_freq": total_freq.tolist(), "freq_matrix": freq_matrix.tolist()}
        with open(run_dir / "routing_freq.json", "w") as f:
            json.dump(freq_info, f, indent=2)

        if is_bridge:
            with open(args.layer_bridge_summary, "r") as f:
                layer_summary = json.load(f)
            tau_disp = resolve_tau_disp(args.tau_disp, layer_summary)
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
            meta["constraint_method"] = "bridge_admissibility"
            meta["tau_disp"] = str(args.tau_disp)
            meta["tau_disp_resolved"] = tau_disp
        else:
            groups = mmc.hierarchical_cluster_groups(freq_matrix, n_clusters=target_n, linkage_method=args.linkage)
            meta["constraint_method"] = "none"

        meta["method"] = tag
        meta["merge_weighting"] = "frequency_weighted"
        meta["probe_count"] = args.probe_count
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save merge plan
    mmc.save_merge_plan(run_dir / "merge_plan.json", groups, meta)
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[CPU-Merge] Merge plan: {len(groups)} groups")

    # --- Step 3: Apply merge on CPU (layer by layer) ---
    print("[CPU-Merge] Applying merge on CPU (layer by layer)...")
    _apply_merge_cpu(lang, groups, merge_weights)

    # --- Step 4: Save merged model ---
    save_path = run_dir / "merged_model"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"[CPU-Merge] Saving merged model to {save_path}...")
    vl_gpt.save_pretrained(str(save_path), safe_serialization=True)
    proc = DeepseekVLV2Processor.from_pretrained(args.model_path)
    proc.save_pretrained(str(save_path))

    print(f"[CPU-Merge] Done. n_routed {n_old} -> {len(groups)}. Output: {run_dir}")


def _build_probes(hidden_size, count, seed, device, dtype, scale):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(count, hidden_size, generator=gen, dtype=torch.float32)
    x = x * (scale / math.sqrt(float(hidden_size)))
    return x.to(device=device, dtype=dtype)


def _output_signatures(moes, probe_inputs, batch_size):
    n_experts = len(moes[0].experts)
    sigs = []
    batches = list(torch.split(probe_inputs, batch_size, dim=0))
    with torch.no_grad():
        for eidx in range(n_experts):
            layer_acc = []
            for moe in moes:
                parts = []
                for pb in batches:
                    parts.append(moe.experts[eidx](pb).float().cpu())
                layer_acc.append(torch.cat(parts, dim=0))
            mean_out = torch.stack(layer_acc, dim=0).mean(dim=0)
            sigs.append(mean_out.reshape(-1).numpy())
    return np.stack(sigs, axis=0)


def _collect_routing_freq(moes, probe_inputs, batch_size, topk):
    """Collect routing frequencies by passing probes through each MoE gate.
    
    The gate expects 3D input [bsz, seq_len, hidden_size] and returns
    (topk_idx, topk_weight, aux_loss).
    """
    n_experts = len(moes[0].experts)
    n_layers = len(moes)
    freq = np.zeros((n_experts, n_layers), dtype=np.float64)
    batches = list(torch.split(probe_inputs, batch_size, dim=0))
    total_tokens = 0
    with torch.no_grad():
        for li, moe in enumerate(moes):
            gate = moe.gate
            for pb in batches:
                pb_3d = pb.unsqueeze(1)  # [batch, 1, hidden] — single-token sequences
                topk_idx, topk_weight, aux_loss = gate(pb_3d)
                sel = topk_idx.reshape(-1, topk_idx.shape[-1])  # [batch, topk]
                for ei in range(n_experts):
                    freq[ei, li] += (sel == ei).sum().item()
                if li == 0:
                    total_tokens += sel.shape[0]
    if total_tokens > 0:
        freq /= total_tokens
    total = freq.sum(axis=1)
    s = total.sum()
    if s > 0:
        total /= s
    return freq, total


def _apply_merge_cpu(language_model, groups, freq_weights=None):
    """Apply merge partition entirely on CPU, layer by layer."""
    from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE, DeepseekV2MLP, MoEGate

    cfg = language_model.config
    new_n = len(groups)
    cfg.n_routed_experts = new_n

    for layer_idx, layer in enumerate(language_model.model.layers):
        if not isinstance(layer.mlp, DeepseekV2MoE):
            continue

        old_moe = layer.mlp
        print(f"  Layer {layer_idx}: merging {len(old_moe.experts)} -> {new_n} experts on CPU")

        new_moe = DeepseekV2MoE(cfg)

        # Copy shared experts
        if old_moe.config.n_shared_experts is not None and new_moe.shared_experts is not None:
            new_moe.shared_experts.load_state_dict(old_moe.shared_experts.state_dict())

        # Merge experts
        for ni, g in enumerate(groups):
            mlps = [old_moe.experts[i] for i in g]
            if freq_weights is not None:
                w = freq_weights[list(g)]
                ws = w.sum()
                w = w / ws if ws > 0 else np.ones(len(g)) / len(g)
                avg_sd = _freq_weighted_avg_sd(mlps, w)
            else:
                avg_sd = _simple_avg_sd(mlps)
            new_moe.experts[ni].load_state_dict(avg_sd)

        # Merge gate
        _merge_gate(old_moe.gate, groups, new_moe.gate, freq_weights)

        new_moe.eval()
        layer.mlp = new_moe

        del old_moe
        gc.collect()

    print(f"[CPU-Merge] All MoE layers merged. n_routed_experts = {new_n}")


def _simple_avg_sd(mlps):
    ref = mlps[0].state_dict()
    out = {}
    for k in ref:
        stacked = torch.stack([m.state_dict()[k].float() for m in mlps], dim=0)
        out[k] = stacked.mean(0).to(ref[k].dtype)
    return out


def _freq_weighted_avg_sd(mlps, w):
    ref = mlps[0].state_dict()
    out = {}
    for k in ref:
        stacked = torch.stack([m.state_dict()[k].float() for m in mlps], dim=0)
        wt = torch.tensor(w, dtype=torch.float32).reshape(-1, *([1] * (stacked.dim() - 1)))
        out[k] = (stacked * wt).sum(0).to(ref[k].dtype)
    return out


def _merge_gate(old_gate, groups, new_gate, freq_weights=None):
    with torch.no_grad():
        ow = old_gate.weight
        nw = new_gate.weight
        for ni, g in enumerate(groups):
            idx = list(g)
            if freq_weights is not None:
                w = freq_weights[idx]
                ws = w.sum()
                w = w / ws if ws > 0 else np.ones(len(idx)) / len(idx)
                wt = torch.tensor(w, dtype=torch.float32).reshape(-1, 1)
                nw[ni].copy_((ow[idx].float() * wt).sum(0).to(nw.dtype))
            else:
                nw[ni].copy_(ow[idx].float().mean(0).to(nw.dtype))

        if hasattr(old_gate, "e_score_correction_bias") and hasattr(new_gate, "e_score_correction_bias"):
            ob = old_gate.e_score_correction_bias
            nb = new_gate.e_score_correction_bias
            for ni, g in enumerate(groups):
                idx = list(g)
                if freq_weights is not None:
                    w = freq_weights[idx]
                    ws = w.sum()
                    w = w / ws if ws > 0 else np.ones(len(idx)) / len(idx)
                    wt = torch.tensor(w, dtype=torch.float32)
                    nb[ni].copy_((ob[idx].float() * wt).sum().to(nb.dtype))
                else:
                    nb[ni].copy_(ob[idx].float().mean().to(nb.dtype))


if __name__ == "__main__":
    main()
