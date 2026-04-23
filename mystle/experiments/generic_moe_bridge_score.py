#!/usr/bin/env python3
"""Generic multimodal MoE bridge-score probe across model families."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import percentileofscore, spearmanr
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generic_moe_bridge_score")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--calibration-dir", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--num-subsets", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau-disp", type=float, default=0.0)
    p.add_argument("--load-in-4bit", action="store_true")
    return p.parse_args()


def load_manifest(cal_dir: Path) -> List[Dict[str, Any]]:
    for name in ("manifest.jsonl", "manifest.json"):
        path = cal_dir / name
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            rows: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    raise FileNotFoundError(f"No manifest found in {cal_dir}")


def extract_layer_idx(name: str) -> int:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def register_moe_hooks(model: nn.Module) -> Tuple[List[Any], Dict[int, Dict[str, torch.Tensor | None]]]:
    handles: List[Any] = []
    captures: Dict[int, Dict[str, torch.Tensor | None]] = {}
    for name, mod in model.named_modules():
        if not (hasattr(mod, "experts") and hasattr(mod, "gate")):
            continue
        if not isinstance(getattr(mod, "experts", None), nn.ModuleList):
            continue
        if len(mod.experts) <= 1:
            continue
        lid = extract_layer_idx(name)
        if lid < 0:
            continue
        stor: Dict[str, torch.Tensor | None] = {"topk": None, "moe_out": None}

        def gate_hook(_m, _inp, out, s=stor):
            if isinstance(out, tuple):
                s["topk"] = out[0].detach().cpu()
            elif torch.is_tensor(out):
                s["topk"] = out.detach().cpu()

        def moe_hook(_m, _inp, out, s=stor):
            if isinstance(out, tuple):
                first = out[0]
                if torch.is_tensor(first):
                    s["moe_out"] = first.detach().cpu()
            elif torch.is_tensor(out):
                s["moe_out"] = out.detach().cpu()

        handles.append(mod.gate.register_forward_hook(gate_hook))
        handles.append(mod.register_forward_hook(moe_hook))
        captures[lid] = stor
    return handles, captures


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_inputs(processor, question: str, image: Image.Image | None) -> Dict[str, Any]:
    if image is not None:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]

    try:
        # Most modern VLM processors support this path.
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if image is not None:
            return processor(text=text, images=[image], return_tensors="pt")
        return processor(text=text, return_tensors="pt")


def pick_image_mask(inputs_vis: Dict[str, Any], target_len: int) -> torch.Tensor | None:
    for k in ("images_seq_mask", "image_seq_mask", "vision_mask", "pixel_attention_mask"):
        if k in inputs_vis and torch.is_tensor(inputs_vis[k]):
            mask = inputs_vis[k].cpu()
            if mask.dim() == 2 and mask.shape[1] >= target_len:
                return mask[:, -target_len:]
    return None


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cal_dir = Path(args.calibration_dir)
    out_dir = Path(args.results_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(cal_dir)
    if args.max_samples > 0:
        manifest = manifest[: args.max_samples]
    n_total = len(manifest)
    logger.info("Loaded %d calibration samples", n_total)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s", args.model_path, device)

    common_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        common_kwargs["load_in_4bit"] = True
    else:
        common_kwargs["torch_dtype"] = torch.bfloat16

    model = None
    model_classes = [AutoModelForCausalLM]
    if AutoModelForImageTextToText is not None:
        model_classes.insert(0, AutoModelForImageTextToText)
    for cls in model_classes:
        try:
            model = cls.from_pretrained(args.model_path, **common_kwargs)
            break
        except Exception as e:
            logger.warning("Load via %s failed: %s", cls.__name__, e)
    if model is None:
        raise RuntimeError(f"Failed to load model {args.model_path}")
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    handles, captures = register_moe_hooks(model)
    moe_layers = sorted(captures.keys())
    if not moe_layers:
        raise RuntimeError("No MoE layers found (module with experts+gate)")
    logger.info("Detected %d MoE layers: %s", len(moe_layers), moe_layers)

    n_experts = None
    for mod in model.modules():
        if hasattr(mod, "experts") and isinstance(getattr(mod, "experts", None), nn.ModuleList):
            if len(mod.experts) > 1:
                n_experts = len(mod.experts)
                break
    if n_experts is None:
        raise RuntimeError("Unable to infer n_experts")
    logger.info("n_experts=%d", n_experts)

    n_sub = args.num_subsets
    stats = {
        lid: {
            "sum_vis": np.zeros((n_sub, n_experts), dtype=np.float64),
            "sum_txt": np.zeros((n_sub, n_experts), dtype=np.float64),
            "cnt_vis": np.zeros((n_sub, n_experts), dtype=np.int64),
            "cnt_txt": np.zeros((n_sub, n_experts), dtype=np.int64),
        }
        for lid in moe_layers
    }

    success = 0
    for si, sample in enumerate(manifest):
        subset = min(si // max(1, n_total // n_sub), n_sub - 1)
        question = sample.get("question", sample.get("text", "Describe this image."))
        img_file = sample.get("image", sample.get("image_path", ""))
        img_path = cal_dir / "images" / img_file if img_file else None
        if img_path is None or not img_path.exists():
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        try:
            inputs_vis = to_device(build_inputs(processor, question, image), device)
            for s in captures.values():
                s["topk"] = None
                s["moe_out"] = None
            with torch.no_grad():
                _ = model(**inputs_vis, use_cache=False)
            vis_data = {
                lid: {"topk": captures[lid]["topk"], "moe_out": captures[lid]["moe_out"]}
                for lid in moe_layers
                if captures[lid]["topk"] is not None and captures[lid]["moe_out"] is not None
            }

            inputs_txt = to_device(build_inputs(processor, question, None), device)
            for s in captures.values():
                s["topk"] = None
                s["moe_out"] = None
            with torch.no_grad():
                _ = model(**inputs_txt, use_cache=False)
            txt_data = {
                lid: {"topk": captures[lid]["topk"], "moe_out": captures[lid]["moe_out"]}
                for lid in moe_layers
                if captures[lid]["topk"] is not None and captures[lid]["moe_out"] is not None
            }

            for lid in moe_layers:
                if lid not in vis_data or lid not in txt_data:
                    continue
                moe_v = vis_data[lid]["moe_out"].float()
                moe_t = txt_data[lid]["moe_out"].float()
                topk_v = vis_data[lid]["topk"]
                sv, st = moe_v.shape[1], moe_t.shape[1]
                L = min(sv, st)
                if L <= 0:
                    continue
                moe_v = moe_v[:, -L:, :]
                moe_t = moe_t[:, -L:, :]
                if topk_v.dim() == 2:
                    topk = topk_v.view(1, sv, -1)[:, -L:, :]
                else:
                    topk = topk_v[:, -L:, :]

                diff = torch.norm(moe_v - moe_t, dim=-1)
                img_mask = pick_image_mask(inputs_vis, L)
                if img_mask is not None:
                    vis_tokens = img_mask[0].bool()
                    txt_tokens = ~vis_tokens
                else:
                    # Fallback if model doesn't expose image token mask
                    vis_tokens = torch.zeros(L, dtype=torch.bool)
                    txt_tokens = torch.ones(L, dtype=torch.bool)

                st_dict = stats[lid]
                for ti in range(L):
                    d = float(diff[0, ti].item())
                    for e in topk[0, ti].unique():
                        ei = int(e.item())
                        if not (0 <= ei < n_experts):
                            continue
                        if vis_tokens[ti]:
                            st_dict["sum_vis"][subset, ei] += d
                            st_dict["cnt_vis"][subset, ei] += 1
                        if txt_tokens[ti]:
                            st_dict["sum_txt"][subset, ei] += d
                            st_dict["cnt_txt"][subset, ei] += 1
            success += 1
        except Exception as e:
            if si < 5:
                logger.warning("Sample %d failed: %s", si, e)
            continue
        if (si + 1) % 16 == 0:
            logger.info("Processed %d/%d (success=%d)", si + 1, n_total, success)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    layers_out: Dict[str, Dict[str, Any]] = {}
    for lid in moe_layers:
        st = stats[lid]
        sum_vis = st["sum_vis"].sum(0)
        sum_txt = st["sum_txt"].sum(0)
        cnt_vis = st["cnt_vis"].sum(0).astype(np.float64)
        cnt_txt = st["cnt_txt"].sum(0).astype(np.float64)
        Iv = safe_div(sum_vis, cnt_vis)
        It = safe_div(sum_txt, cnt_txt)
        B = Iv - It
        eps = 1e-8
        M = (It - Iv) / (It + Iv + eps)
        valid = np.isfinite(B)
        if valid.sum() > 0:
            Bv = B[valid]
            bar = float(Bv.mean())
            sigma = float(Bv.std(ddof=0))
            Bz = ((B - bar) / (sigma + eps)).tolist()
            Bz = [float(x) if np.isfinite(x) else float("nan") for x in Bz]
        else:
            bar, sigma = float("nan"), float("nan")
            Bz = [float("nan")] * n_experts
        layers_out[str(lid)] = {
            "layer_idx": int(lid),
            "bar_B_l": bar,
            "sigma_l": sigma,
            "sigma_l_ge_tau_disp": bool(np.isfinite(sigma) and sigma >= args.tau_disp),
            "n_experts": n_experts,
            "B": [float(x) if np.isfinite(x) else float("nan") for x in B],
            "M": [float(x) if np.isfinite(x) else float("nan") for x in M],
            "B_z": Bz,
        }

    abs_list = [abs(layers_out[str(l)]["bar_B_l"]) for l in moe_layers if np.isfinite(layers_out[str(l)]["bar_B_l"])]
    for lid in moe_layers:
        key = str(lid)
        b = layers_out[key]["bar_B_l"]
        if abs_list and np.isfinite(b):
            layers_out[key]["S_l_sensitivity"] = float(percentileofscore(abs_list, abs(b), kind="rank") / 100.0)
        else:
            layers_out[key]["S_l_sensitivity"] = float("nan")

    stability = {}
    for lid in moe_layers:
        st = stats[lid]
        Iv_sub = safe_div(st["sum_vis"], st["cnt_vis"].astype(np.float64))
        It_sub = safe_div(st["sum_txt"], st["cnt_txt"].astype(np.float64))
        B_sub = Iv_sub - It_sub
        pairs = []
        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                v1, v2 = B_sub[i], B_sub[j]
                mask = np.isfinite(v1) & np.isfinite(v2)
                if mask.sum() < 4:
                    pairs.append({"pair": [i, j], "spearman": None})
                else:
                    rho, _ = spearmanr(v1[mask], v2[mask])
                    pairs.append({"pair": [i, j], "spearman": float(rho)})
        stability[str(lid)] = pairs

    summary = {
        "meta": {
            "model_path": args.model_path,
            "n_samples_success": success,
            "n_samples_total": n_total,
            "tau_disp": args.tau_disp,
            "run_id": args.run_id,
        },
        "moe_layer_indices": [int(x) for x in moe_layers],
        "layers": layers_out,
        "stability_spearman": stability,
    }
    with (out_dir / "layer_bridge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "bridge_score_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved bridge score summary to %s", out_dir)


if __name__ == "__main__":
    main()

