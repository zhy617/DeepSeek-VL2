#!/usr/bin/env python3
"""
Bridge score computation for Kimi-VL-A3B.

Adapts bridge_score.py for Kimi-VL's architecture:
  - Uses AutoProcessor for tokenization
  - Detects DeepseekV3MoE layers via generic attribute probing
  - Outputs layer_bridge_summary.json in the same format as bridge_score.py

Methodology (same as original):
  For each calibration sample, run forward passes with:
    1. Original image + text  (I_vis)
    2. Text-only / ablated image  (I_txt)
  Compare MoE routing decisions to compute per-expert bridge scores.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("kimivl_bridge_score")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bridge score for Kimi-VL-A3B MoE.")
    p.add_argument("--model-path", type=str, default="moonshotai/Kimi-VL-A3B-Thinking")
    p.add_argument("--calibration-dir", type=str, required=True)
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--num-subsets", type=int, default=3)
    p.add_argument("--tau-disp", type=float, default=0.0)
    p.add_argument("--layer-summary-eps", type=float, default=1e-8)
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def iter_moe_modules(language_model) -> List[nn.Module]:
    moes = []
    for layer in language_model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            moes.append(mlp)
    return moes


def get_moe_layer_indices(language_model) -> List[int]:
    indices = []
    for i, layer in enumerate(language_model.model.layers):
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts"):
            indices.append(i)
    return indices


def load_calibration(calib_dir: str, max_samples: int = -1) -> List[Dict]:
    manifest_path = Path(calib_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {calib_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    samples = manifest if isinstance(manifest, list) else manifest.get("samples", [])
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


class MoERoutingHook:
    """Hook to capture routing decisions at each MoE layer."""

    def __init__(self, moes: List[nn.Module]):
        self.moes = moes
        self.routing_data: List[Optional[torch.Tensor]] = [None] * len(moes)
        self._handles = []

    def install(self):
        for i, moe in enumerate(self.moes):
            handle = moe.gate.register_forward_hook(self._make_hook(i))
            self._handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            topk_idx = output[0]
            self.routing_data[layer_idx] = topk_idx.detach().cpu()
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self.routing_data = [None] * len(self.moes)


def compute_expert_importance(
    routing_data: List[Optional[torch.Tensor]],
    n_experts: int,
) -> List[np.ndarray]:
    """Convert routing decisions to per-expert importance vectors."""
    result = []
    for topk_idx in routing_data:
        if topk_idx is None:
            result.append(np.zeros(n_experts, dtype=np.float64))
            continue
        flat = topk_idx.reshape(-1)
        counts = np.zeros(n_experts, dtype=np.float64)
        for ei in range(n_experts):
            counts[ei] = (flat == ei).sum().item()
        total = flat.shape[0]
        if total > 0:
            counts /= total
        result.append(counts)
    return result


def safe_div(a, b):
    out = np.full_like(a, np.nan, dtype=np.float64)
    m = b > 0
    out[m] = a[m] / b[m]
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.results_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Kimi-VL model: %s", args.model_path)
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True,
    )

    lang = model.language_model
    moes = iter_moe_modules(lang)
    moe_indices = get_moe_layer_indices(lang)
    n_experts = len(moes[0].experts)
    n_moe_layers = len(moes)

    logger.info("Found %d MoE layers with %d experts each", n_moe_layers, n_experts)

    # Load calibration data
    samples = load_calibration(args.calibration_dir, args.max_samples)
    n_total = len(samples)
    logger.info("Loaded %d calibration samples", n_total)

    # Split into subsets for stability analysis
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    subset_size = n_total // args.num_subsets
    subsets = [indices[i * subset_size:(i + 1) * subset_size] for i in range(args.num_subsets)]

    # Accumulators: [n_subsets, n_moe_layers, n_experts]
    sum_vis = np.zeros((args.num_subsets, n_moe_layers, n_experts), dtype=np.float64)
    sum_txt = np.zeros((args.num_subsets, n_moe_layers, n_experts), dtype=np.float64)
    cnt_vis = np.zeros((args.num_subsets, n_moe_layers, n_experts), dtype=np.int64)
    cnt_txt = np.zeros((args.num_subsets, n_moe_layers, n_experts), dtype=np.int64)

    hook = MoERoutingHook(moes)
    hook.install()

    calib_dir = Path(args.calibration_dir)

    for sample_idx in tqdm(range(n_total), desc="bridge_score"):
        sample = samples[sample_idx]
        subset_ids = [si for si in range(args.num_subsets) if sample_idx in subsets[si]]

        img_path = calib_dir / sample.get("image", sample.get("image_path", ""))
        question = sample.get("question", sample.get("text", "Describe this image."))

        if not img_path.exists():
            continue

        try:
            img = Image.open(str(img_path)).convert("RGB")
        except Exception:
            continue

        # --- Forward with image + text ---
        hook.clear()
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs, use_cache=False)
            vis_importance = compute_expert_importance(hook.routing_data, n_experts)
        except Exception as e:
            logger.warning("Sample %d vis forward failed: %s", sample_idx, e)
            continue

        # --- Forward with text only ---
        hook.clear()
        try:
            messages_txt = [{"role": "user", "content": [
                {"type": "text", "text": question},
            ]}]
            inputs_txt = processor.apply_chat_template(
                messages_txt, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            inputs_txt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_txt.items()}
            with torch.no_grad():
                model(**inputs_txt, use_cache=False)
            txt_importance = compute_expert_importance(hook.routing_data, n_experts)
        except Exception as e:
            logger.warning("Sample %d txt forward failed: %s", sample_idx, e)
            continue

        # Accumulate
        for si in subset_ids:
            for li in range(n_moe_layers):
                sum_vis[si, li] += vis_importance[li]
                sum_txt[si, li] += txt_importance[li]
                cnt_vis[si, li] += 1
                cnt_txt[si, li] += 1

        if sample_idx % 50 == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    hook.remove()

    # Compute bridge scores: B(e,l) = I_vis(e,l) - I_txt(e,l)
    mean_vis = safe_div(sum_vis, cnt_vis)  # [subsets, layers, experts]
    mean_txt = safe_div(sum_txt, cnt_txt)
    B = mean_vis - mean_txt  # Bridge score: positive = visual-dominant

    # Pool across subsets
    B_pooled = np.nanmean(B, axis=0)  # [layers, experts]

    # Compute layer-level summary
    layer_summary = {
        "meta": {
            "tau_disp": args.tau_disp,
            "eps": args.layer_summary_eps,
            "description": "bar_B_l/sigma_l from pooled B. B_z = per-expert z-score.",
        },
        "moe_layer_indices": moe_indices,
        "layers": {},
    }

    for li in range(n_moe_layers):
        b_layer = B_pooled[li]
        valid = ~np.isnan(b_layer)
        if not valid.any():
            continue

        b_valid = b_layer[valid]
        bar_B = float(np.mean(b_valid))
        sigma = float(np.std(b_valid)) + args.layer_summary_eps

        b_z = np.full(n_experts, 0.0)
        b_z[valid] = (b_valid - bar_B) / sigma

        # Subset stability (Spearman correlation)
        spearman_vals = []
        for s1 in range(args.num_subsets):
            for s2 in range(s1 + 1, args.num_subsets):
                b1 = B[s1, li]
                b2 = B[s2, li]
                v = ~np.isnan(b1) & ~np.isnan(b2)
                if v.sum() > 2:
                    rho, _ = spearmanr(b1[v], b2[v])
                    if np.isfinite(rho):
                        spearman_vals.append(rho)

        layer_key = str(moe_indices[li])
        layer_summary["layers"][layer_key] = {
            "bar_B_l": bar_B,
            "sigma_l": sigma,
            "B_z": b_z.tolist(),
            "B_raw": b_layer.tolist(),
            "spearman_stability": float(np.mean(spearman_vals)) if spearman_vals else None,
            "n_experts": n_experts,
        }

    # Save results
    summary_path = out_dir / "layer_bridge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(layer_summary, f, ensure_ascii=False, indent=2)
    logger.info("Layer bridge summary saved: %s", summary_path)

    # Save detailed results
    detailed = {
        "model_path": args.model_path,
        "model_family": "kimi_vl",
        "n_moe_layers": n_moe_layers,
        "n_experts": n_experts,
        "moe_layer_indices": moe_indices,
        "n_samples": n_total,
        "num_subsets": args.num_subsets,
        "B_pooled": B_pooled.tolist(),
    }
    detail_path = out_dir / "bridge_score_results.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)
    logger.info("Detailed results saved: %s", detail_path)

    # Print summary
    logger.info("=== Bridge Score Summary ===")
    for lk, row in layer_summary["layers"].items():
        stab = row.get("spearman_stability")
        stab_str = f"{stab:.3f}" if stab is not None else "N/A"
        logger.info(
            "  Layer %s: bar_B=%.4f, sigma=%.4f, stability=%s",
            lk, row["bar_B_l"], row["sigma_l"], stab_str,
        )


if __name__ == "__main__":
    main()
