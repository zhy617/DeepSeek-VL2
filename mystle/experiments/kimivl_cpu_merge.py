#!/usr/bin/env python3
"""
CPU-based merge for Kimi-VL-A3B (DeepseekV3 MoE backbone).

Adapts cpu_merge_apply.py logic for Kimi-VL's architecture:
  - language_model (not language) attribute
  - DeepseekV3MoE / DeepseekV3MLP classes from trust_remote_code
  - AutoProcessor for tokenizer/image saving
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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU-based merge for Kimi-VL-A3B (DeepseekV3 MoE)."
    )
    p.add_argument("--model-path", type=str, default="moonshotai/Kimi-VL-A3B-Thinking")
    p.add_argument(
        "--output-dir", type=str,
        default=os.path.expanduser("~/fsas/projects/bridge-vlm/results/kimivl"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--method", type=str, required=True,
        choices=("hcsmoe", "admissibility"),
        help="Merge method: hcsmoe (baseline) or admissibility (bridge-constrained)",
    )
    p.add_argument("--compression-keep", type=float, default=0.5)
    p.add_argument("--linkage", type=str, default="average")
    p.add_argument(
        "--layer-bridge-summary", type=str, default="",
        help="Path to layer_bridge_summary.json (required for admissibility)",
    )
    p.add_argument("--tau-disp", type=str, default="auto_p75")
    p.add_argument("--use-bz-quantile", action="store_true", default=True)
    p.add_argument("--bz-quantile-for-cutoff", type=float, default=0.75)
    p.add_argument("--delta-affinity", type=float, default=2.0)
    p.add_argument("--admissibility-scope", type=str, default="max_sigma_layer")
    p.add_argument("--forbidden-penalty", type=float, default=10.0)
    return p.parse_args()


# ── MoE utilities (generic, no deepseek_vl2 import needed) ──


def iter_moe_modules(language_model):
    """Find all MoE layers in a DeepseekV3-style language model."""
    moes = []
    for layer in language_model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            moes.append(mlp)
    return moes


def expert_weight_vector(expert) -> np.ndarray:
    with torch.no_grad():
        parts = [
            expert.gate_proj.weight.flatten(),
            expert.up_proj.weight.flatten(),
            expert.down_proj.weight.flatten(),
        ]
        return torch.cat(parts).float().cpu().numpy()


def aggregate_expert_vectors(moes) -> np.ndarray:
    n_experts = len(moes[0].experts)
    vecs = []
    for eidx in range(n_experts):
        parts = []
        for moe in moes:
            parts.append(expert_weight_vector(moe.experts[eidx]))
        vecs.append(np.concatenate(parts))
    return np.stack(vecs, axis=0)


def hierarchical_cluster_groups(X, n_clusters, linkage_method="average"):
    dist_condensed = pdist(X, metric="euclidean")
    Z = linkage(dist_condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    groups = defaultdict(list)
    for idx, lab in enumerate(labels):
        groups[lab].append(idx)
    return [sorted(v) for v in groups.values()]


def hierarchical_cluster_from_full_distance(D, n_clusters, linkage_method="average"):
    dist_condensed = squareform(D, checks=False)
    Z = linkage(dist_condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    groups = defaultdict(list)
    for idx, lab in enumerate(labels):
        groups[lab].append(idx)
    return [sorted(v) for v in groups.values()]


def constrained_distance_matrix(
    X, n_experts, layer_summary,
    tau_disp=0.0, tau_z=None, delta_affinity=2.0,
    use_per_layer_tau_z=True, bz_quantile_for_cutoff=0.75,
    forbidden_penalty=10.0, admissibility_scope="max_sigma_layer",
):
    dist_condensed = pdist(X, metric="euclidean")
    D = squareform(dist_condensed)
    layers = layer_summary.get("layers", {})
    if admissibility_scope == "max_sigma_layer":
        best_key = max(layers, key=lambda k: abs(layers[k].get("sigma_l", 0)))
        layer_keys = [best_key]
    else:
        layer_keys = list(layers.keys())

    for lk in layer_keys:
        row = layers[lk]
        sigma_l = row.get("sigma_l", 0)
        if abs(sigma_l) < tau_disp:
            continue
        bz = np.array(row.get("B_z", [0.0] * n_experts))
        if use_per_layer_tau_z:
            tau_z_local = float(np.quantile(np.abs(bz), bz_quantile_for_cutoff))
        else:
            tau_z_local = tau_z if tau_z is not None else 0.0
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                bi, bj = bz[i], bz[j]
                if abs(bi) > tau_z_local and abs(bj) > tau_z_local:
                    if bi * bj < 0:
                        D[i, j] += forbidden_penalty
                        D[j, i] += forbidden_penalty
                    elif abs(bi - bj) > delta_affinity:
                        D[i, j] += forbidden_penalty * 0.5
                        D[j, i] += forbidden_penalty * 0.5
    return D


def resolve_target_n(n_old, keep, n_group, topk):
    raw = max(1, round(n_old * keep))
    if n_group and n_group > 1:
        raw = max(n_group, (raw // n_group) * n_group)
    return max(topk, raw)


def resolve_tau_disp(tau_disp_str, layer_summary):
    if tau_disp_str.startswith("auto"):
        layers = layer_summary.get("layers", {})
        sigmas = [row["sigma_l"] for row in layers.values()
                  if "sigma_l" in row and np.isfinite(row["sigma_l"])]
        if not sigmas:
            return 0.0
        q = int(tau_disp_str.split("_p")[1]) / 100.0 if "_p" in tau_disp_str else 0.5
        return float(np.quantile(sigmas, q))
    return float(tau_disp_str)


def save_merge_plan(path, groups, meta):
    plan = {"groups": [sorted(g) for g in groups], "meta": meta}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)


def run_id_for(method, keep):
    keep_str = f"{keep:.2f}".replace(".", "p").rstrip("0").rstrip("p") or "0"
    return f"baselines/{method}_keep_{keep_str}"


# ── Merge application ──


def simple_avg_sd(mlps):
    ref = mlps[0].state_dict()
    out = {}
    for k in ref:
        stacked = torch.stack([m.state_dict()[k].float() for m in mlps], dim=0)
        out[k] = stacked.mean(0).to(ref[k].dtype)
    return out


def merge_gate(old_gate, groups, new_gate):
    with torch.no_grad():
        ow = old_gate.weight
        nw = new_gate.weight
        for ni, g in enumerate(groups):
            idx = list(g)
            nw[ni].copy_(ow[idx].float().mean(0).to(nw.dtype))
        if hasattr(old_gate, "e_score_correction_bias") and hasattr(new_gate, "e_score_correction_bias"):
            ob = old_gate.e_score_correction_bias
            nb = new_gate.e_score_correction_bias
            for ni, g in enumerate(groups):
                nb[ni].copy_(ob[list(g)].float().mean().to(nb.dtype))


def apply_merge_cpu(language_model, groups, MoEClass):
    cfg = language_model.config
    new_n = len(groups)
    cfg.n_routed_experts = new_n

    for layer_idx, layer in enumerate(language_model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "gate") or not hasattr(mlp, "experts"):
            continue
        print(f"  Layer {layer_idx}: merging {len(mlp.experts)} -> {new_n} experts on CPU")
        new_moe = MoEClass(cfg)
        if mlp.config.n_shared_experts is not None and new_moe.shared_experts is not None:
            new_moe.shared_experts.load_state_dict(mlp.shared_experts.state_dict())
        for ni, g in enumerate(groups):
            avg_sd = simple_avg_sd([mlp.experts[i] for i in g])
            new_moe.experts[ni].load_state_dict(avg_sd)
        merge_gate(mlp.gate, groups, new_moe.gate)
        new_moe.eval()
        layer.mlp = new_moe
        del mlp
        gc.collect()

    print(f"[Merge] All MoE layers merged. n_routed_experts = {new_n}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_root = Path(args.output_dir).expanduser().resolve()
    run_id = run_id_for(args.method, args.compression_keep)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[KimiVL-Merge] method={args.method}, keep={args.compression_keep}")
    print(f"[KimiVL-Merge] Output: {run_dir}")
    print(f"[KimiVL-Merge] Loading model on CPU (bf16): {args.model_path}")

    from transformers import AutoModelForCausalLM, AutoProcessor

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    vl_gpt = vl_gpt.eval()
    lang = vl_gpt.language_model
    print("[KimiVL-Merge] Model loaded on CPU")

    moes = iter_moe_modules(lang)
    if not moes:
        raise RuntimeError("No MoE layers found in Kimi-VL language model")
    print(f"[KimiVL-Merge] Found {len(moes)} MoE layers")

    MoEClass = type(moes[0])
    cfg = lang.config
    n_old = int(cfg.n_routed_experts)
    n_group = getattr(cfg, "n_group", None)
    topk = int(cfg.num_experts_per_tok)
    target_n = resolve_target_n(n_old, args.compression_keep, n_group, topk)

    print(f"[KimiVL-Merge] n_routed: {n_old} -> {target_n}")

    meta = {
        "model_path": args.model_path,
        "model_family": "kimi_vl",
        "n_routed_old": n_old,
        "n_routed_new": target_n,
        "compression_keep": args.compression_keep,
        "linkage": args.linkage,
        "seed": args.seed,
        "cpu_merge": True,
    }

    if args.method == "hcsmoe":
        print("[KimiVL-Merge] HC-SMoE: weight-space clustering")
        X = aggregate_expert_vectors(moes)
        groups = hierarchical_cluster_groups(X, n_clusters=target_n, linkage_method=args.linkage)
        meta["method"] = "HC-SMoE"

    elif args.method == "admissibility":
        if not args.layer_bridge_summary:
            raise ValueError("--layer-bridge-summary required for admissibility method")
        print("[KimiVL-Merge] Admissibility: bridge-constrained clustering")
        with open(args.layer_bridge_summary, "r") as f:
            layer_summary = json.load(f)
        tau_disp = resolve_tau_disp(args.tau_disp, layer_summary)
        X = aggregate_expert_vectors(moes)
        D = constrained_distance_matrix(
            X, n_old, layer_summary,
            tau_disp=tau_disp,
            tau_z=None if args.use_bz_quantile else 0.0,
            delta_affinity=args.delta_affinity,
            use_per_layer_tau_z=args.use_bz_quantile,
            bz_quantile_for_cutoff=args.bz_quantile_for_cutoff,
            forbidden_penalty=args.forbidden_penalty,
            admissibility_scope=args.admissibility_scope,
        )
        groups = hierarchical_cluster_from_full_distance(D, target_n, args.linkage)
        meta["method"] = "HC-SMoE+Bridge (admissibility)"
        meta["tau_disp"] = str(args.tau_disp)
        meta["tau_disp_resolved"] = tau_disp
        meta["admissibility_scope"] = args.admissibility_scope
    else:
        raise ValueError(f"Unknown method: {args.method}")

    save_merge_plan(run_dir / "merge_plan.json", groups, meta)
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[KimiVL-Merge] Merge plan: {len(groups)} groups")

    print("[KimiVL-Merge] Applying merge on CPU (layer by layer)...")
    apply_merge_cpu(lang, groups, MoEClass)

    save_path = run_dir / "merged_model"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"[KimiVL-Merge] Saving merged model to {save_path}...")
    vl_gpt.save_pretrained(str(save_path), safe_serialization=True)
    try:
        proc = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        proc.save_pretrained(str(save_path))
    except Exception as e:
        print(f"[KimiVL-Merge] Warning: could not save processor: {e}")

    print(f"[KimiVL-Merge] Done. n_routed {n_old} -> {len(groups)}. Output: {run_dir}")


if __name__ == "__main__":
    main()
