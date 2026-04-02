#!/usr/bin/env python3
"""
Step 3: Bridge Score 与 Modality Affinity 计算。

依据 mystle/prompt/run_experiment.md：
- I_visual / I_text / I_mismatch：在原始 forward 下 expert 位于 top-K 的 token 上，
  对 MoE 子层输出 h（DeepseekV2MoE 的 forward 输出，含 shared experts）与干预 forward 的 L2 差期望。
- 为对齐序列，聚合位置限定为「图像 token」：images_seq_mask 与 attention_mask 同时为真
  （text_ablated 仅改文本时，图像区长度与 original 一致）。
- mismatch 图像 resize 到当前样本 manifest 中的 image_size，与 original 同分辨率以便对齐。

输出：带时间戳的 JSON + matplotlib/seaborn 图到 --results-dir。
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
from scipy.stats import percentileofscore, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bridge_score")


def _empty_layer_stats(n_experts: int, n_subsets: int) -> Dict[str, Any]:
    return {
        "sum_vis": np.zeros((n_subsets, n_experts), dtype=np.float64),
        "sum_txt": np.zeros((n_subsets, n_experts), dtype=np.float64),
        "sum_mm": np.zeros((n_subsets, n_experts), dtype=np.float64),
        "cnt_vis": np.zeros((n_subsets, n_experts), dtype=np.int64),
        "cnt_txt": np.zeros((n_subsets, n_experts), dtype=np.int64),
        "cnt_mm": np.zeros((n_subsets, n_experts), dtype=np.int64),
    }


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def text_ablation_question(original_q: str, placeholder: str, match_chars: bool) -> str:
    if not match_chars:
        return placeholder
    piece = placeholder if placeholder else "·"
    if len(original_q) <= 0:
        return piece
    reps = (len(original_q) + len(piece) - 1) // len(piece)
    return (piece * reps)[: len(original_q)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bridge score & modality affinity for DeepSeek-VL2 MoE.")
    p.add_argument("--model-path", type=str, default="deepseek-ai/deepseek-vl2-small")
    p.add_argument(
        "--calibration-dir",
        type=str,
        default=os.path.expanduser("~/fsas/datasets/deepseek-vl2-bridge/calibration"),
        help="含 manifest.jsonl 与 images/ 的目录",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=os.path.expanduser("~/fsas/vlm/deepseek-vl2-bridge/results"),
    )
    p.add_argument("--run-id", type=str, default="", help="子目录名；默认时间戳")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1, help="校准集分辨率不一亦可 batchify；OOM 时减小")
    p.add_argument("--max-samples", type=int, default=-1, help="调试：只跑前 N 条，-1 为全部")
    p.add_argument(
        "--text-ablation-string",
        type=str,
        default="·",
        help="text_ablated：占位片段；与 --match-question-chars 联用可逼近原 question 字符长度",
    )
    p.add_argument(
        "--no-match-question-chars",
        action="store_true",
        help="关闭「占位 question 与原始 question 同字符长度」对齐（默认开启对齐）",
    )
    p.add_argument("--affinity-eps", type=float, default=1e-8, help="M(e) 分母 ε")
    p.add_argument("--num-subsets", type=int, default=3, help="校准集划分的子集数（稳定性分析）")
    p.add_argument("--system-prompt", type=str, default="")
    p.add_argument(
        "--seq-align",
        type=str,
        choices=("strict", "suffix"),
        default="suffix",
        help="序列长度不一致时：strict=跳过；suffix=取末尾 min(S) 对齐（多模态下视觉 token 数可能随图像变化）",
    )
    p.add_argument("--skip-plots", action="store_true", help="不绘制图（无 matplotlib 时用）")
    p.add_argument(
        "--tau-disp",
        type=float,
        default=0.0,
        help="写入 layer_bridge_summary.json：若 sigma_l>=该值则标记 sigma_l_ge_tau_disp（供 Step4 admissibility_merge）",
    )
    p.add_argument(
        "--layer-summary-eps",
        type=float,
        default=1e-8,
        help="层内 B_z 分母 ε（与文档中 ε 一致）",
    )
    p.add_argument(
        "--layer-summary-only",
        action="store_true",
        help="仅从已有 bridge_score_results.json 计算 layer_bridge_summary.json，不加载模型、不重跑 forward",
    )
    p.add_argument(
        "--bridge-results-json",
        type=str,
        default="",
        help="与 --layer-summary-only 联用：bridge_score_results.json 路径；默认同目录写出 layer_bridge_summary.json",
    )
    return p.parse_args()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def subset_id(sample_index: int, n_total: int, n_subsets: int) -> int:
    if n_subsets <= 1:
        return 0
    chunk = int(math.ceil(n_total / n_subsets))
    return min(sample_index // chunk, n_subsets - 1)


def extract_layer_idx(name: str) -> int:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return -1


def register_moe_hooks(language_model: nn.Module) -> Tuple[List[Any], Dict[int, Dict[str, Optional[torch.Tensor]]]]:
    handles: List[Any] = []
    captures: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}

    for name, module in language_model.named_modules():
        if not isinstance(module, DeepseekV2MoE):
            continue
        lid = extract_layer_idx(name)
        if lid < 0:
            logger.warning("无法解析 MoE 层号: %s", name)
            continue
        stor: Dict[str, Optional[torch.Tensor]] = {"topk": None, "moe_out": None}

        def gate_hook(m, inp, out, s=stor):
            s["topk"] = out[0].detach()

        def moe_hook(m, inp, out, s=stor):
            s["moe_out"] = out.detach()

        handles.append(module.gate.register_forward_hook(gate_hook))
        handles.append(module.register_forward_hook(moe_hook))
        captures[lid] = stor

    return handles, captures


def clear_captures(captures: Dict[int, Dict[str, Optional[torch.Tensor]]]) -> None:
    for s in captures.values():
        s["topk"] = None
        s["moe_out"] = None


def run_language_core(
    model: nn.Module,
    prepare: Any,
    device: torch.device,
) -> None:
    """仅跑 transformer core（不算 lm_head），触发 MoE hooks。"""
    inputs_embeds = model.prepare_inputs_embeds(
        input_ids=prepare.input_ids,
        images=prepare.images,
        images_seq_mask=prepare.images_seq_mask,
        images_spatial_crop=prepare.images_spatial_crop,
    )
    _ = model.language.model(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare.attention_mask,
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )


def _reshape_topk(topk_cpu: torch.Tensor, b: int, s_len: int, layer_idx: int) -> Optional[torch.Tensor]:
    if topk_cpu.dim() == 2:
        if topk_cpu.shape[0] != b * s_len:
            logger.warning(
                "层 %s topk 展平长度 %s 与 B*S=%s 不一致，跳过",
                layer_idx,
                topk_cpu.shape[0],
                b * s_len,
            )
            return None
        return topk_cpu.view(b, s_len, -1)
    if topk_cpu.shape[:2] != (b, s_len):
        logger.warning("层 %s topk 形状 %s 与 moe [B,S,...]=%s 不一致，跳过", layer_idx, topk_cpu.shape, (b, s_len))
        return None
    return topk_cpu


def _suffix_align_moe(
    moe_o: torch.Tensor,
    moe_a: torch.Tensor,
    topk: torch.Tensor,
    attn_o: torch.Tensor,
    img_o: torch.Tensor,
    attn_a: torch.Tensor,
    img_a: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """取两序列末尾 L=min(So,Sa) 对齐（多模态视觉 token 长度可能随图像变化）。"""
    L = min(moe_o.shape[1], moe_a.shape[1])
    if L <= 0:
        return moe_o, moe_a, topk, attn_o, img_o, attn_a, img_a
    return (
        moe_o[:, -L:, :],
        moe_a[:, -L:, :],
        topk[:, -L:, :],
        attn_o[:, -L:],
        img_o[:, -L:],
        attn_a[:, -L:],
        img_a[:, -L:],
    )


def accumulate_layer_batch(
    *,
    layer_idx: int,
    st: Dict[str, Any],
    topk_cpu: torch.Tensor,
    moe_orig: torch.Tensor,
    moe_alt: torch.Tensor,
    attention_mask_o: torch.Tensor,
    images_seq_mask_o: torch.Tensor,
    attention_mask_a: torch.Tensor,
    images_seq_mask_a: torch.Tensor,
    n_experts: int,
    subset_ids: List[int],
    key: str,
    seq_align: str,
) -> None:
    """
    topk_cpu: MoEGate 输出 [B*S,K] 或 [B,S,K]（来自 original forward）
    moe_*: [B,S,D] CPU
    """
    b, so, _ = moe_orig.shape
    _, sa, _ = moe_alt.shape
    if so != sa:
        if seq_align == "strict":
            logger.warning(
                "层 %s 序列长不一致 orig=%d alt=%d（strict 跳过）",
                layer_idx,
                so,
                sa,
            )
            return
        L = min(so, sa)
        logger.debug("层 %s 后缀对齐 L=%d (orig=%d alt=%d)", layer_idx, L, so, sa)

    topk_2d = _reshape_topk(topk_cpu, b, so, layer_idx)
    if topk_2d is None:
        return

    mo_o, mo_a = moe_orig, moe_alt
    ao, io, aa, ia = attention_mask_o, images_seq_mask_o, attention_mask_a, images_seq_mask_a
    tk = topk_2d

    if mo_o.shape[1] != mo_a.shape[1]:
        mo_o, mo_a, tk, ao, io, aa, ia = _suffix_align_moe(mo_o, mo_a, tk, ao, io, aa, ia)

    b, s_len, _ = mo_o.shape
    if tk.shape[:2] != (b, s_len):
        logger.warning("层 %s 对齐后 topk %s 与 moe %s 不一致，跳过", layer_idx, tk.shape, mo_o.shape)
        return
    device = mo_o.device
    attn_o = ao.to(device).bool()
    img_o = io.to(device).bool()
    attn_a = aa.to(device).bool()
    img_a = ia.to(device).bool()
    valid = attn_o & img_o & attn_a & img_a

    diff = torch.norm((mo_a.float() - mo_o.float()), dim=-1)

    sum_name = {"vis": "sum_vis", "txt": "sum_txt", "mm": "sum_mm"}[key]
    cnt_name = {"vis": "cnt_vis", "txt": "cnt_txt", "mm": "cnt_mm"}[key]

    for bi in range(b):
        sid = subset_ids[bi]
        sel = valid[bi]
        if not sel.any():
            continue
        tk_sel = tk[bi][sel]
        d = diff[bi][sel]
        for j in range(tk_sel.shape[0]):
            experts_here = tk_sel[j].unique()
            for e in experts_here:
                ei = int(e.item())
                if ei < 0 or ei >= n_experts:
                    continue
                st[sum_name][sid, ei] += float(d[j].item())
                st[cnt_name][sid, ei] += 1


def finalize_scores(
    stats: Dict[int, Dict[str, Any]],
    n_experts: int,
    n_subsets: int,
    eps: float,
) -> Dict[str, Any]:
    """返回每层每子集每 expert 的 I_*, B, M 及全量合并。"""
    layers_out: Dict[str, Any] = {}
    moe_layer_indices = sorted(stats.keys())

    for lid in moe_layer_indices:
        st = stats[lid]
        layer_block: Dict[str, Any] = {"layer_idx": lid, "subsets": [], "pooled_all": {}}

        Iv = safe_div(st["sum_vis"], st["cnt_vis"].astype(np.float64))
        It = safe_div(st["sum_txt"], st["cnt_txt"].astype(np.float64))
        Im = safe_div(st["sum_mm"], st["cnt_mm"].astype(np.float64))

        for sid in range(n_subsets):
            B = Im[sid] - (Iv[sid] + It[sid]) / 2.0
            denom_m = It[sid] + Iv[sid] + eps
            M = (It[sid] - Iv[sid]) / denom_m
            layer_block["subsets"].append(
                {
                    "subset": sid,
                    "I_visual": Iv[sid].tolist(),
                    "I_text": It[sid].tolist(),
                    "I_mismatch": Im[sid].tolist(),
                    "B": B.tolist(),
                    "M": M.tolist(),
                    "count_visual": st["cnt_vis"][sid].tolist(),
                    "count_text": st["cnt_txt"][sid].tolist(),
                    "count_mismatch": st["cnt_mm"][sid].tolist(),
                }
            )

        # 全样本池化
        sum_vis = st["sum_vis"].sum(0)
        sum_txt = st["sum_txt"].sum(0)
        sum_mm = st["sum_mm"].sum(0)
        cnt_vis = st["cnt_vis"].sum(0).astype(np.float64)
        cnt_txt = st["cnt_txt"].sum(0).astype(np.float64)
        cnt_mm = st["cnt_mm"].sum(0).astype(np.float64)
        Iv_a = safe_div(sum_vis, cnt_vis)
        It_a = safe_div(sum_txt, cnt_txt)
        Im_a = safe_div(sum_mm, cnt_mm)
        B_a = Im_a - (Iv_a + It_a) / 2.0
        M_a = (It_a - Iv_a) / (It_a + Iv_a + eps)
        layer_block["pooled_all"] = {
            "I_visual": Iv_a.tolist(),
            "I_text": It_a.tolist(),
            "I_mismatch": Im_a.tolist(),
            "B": B_a.tolist(),
            "M": M_a.tolist(),
        }
        layers_out[str(lid)] = layer_block

    # 子集间 B 的 Spearman（每层，仅在两子集 expert 向量均非全 nan 时）
    stability: Dict[str, Any] = {"per_layer": {}}
    for lid in moe_layer_indices:
        st = stats[lid]
        Iv2 = safe_div(st["sum_vis"], st["cnt_vis"].astype(np.float64))
        It2 = safe_div(st["sum_txt"], st["cnt_txt"].astype(np.float64))
        Im2 = safe_div(st["sum_mm"], st["cnt_mm"].astype(np.float64))
        B_mat = Im2 - (Iv2 + It2) / 2.0  # [n_subsets, n_experts]
        pairs = []
        for i in range(n_subsets):
            for j in range(i + 1, n_subsets):
                v1 = B_mat[i]
                v2 = B_mat[j]
                mask = np.isfinite(v1) & np.isfinite(v2)
                if mask.sum() < 4:
                    pairs.append({"pair": [i, j], "spearman": None, "n_valid": int(mask.sum())})
                    continue
                rho, _ = spearmanr(v1[mask], v2[mask])
                pairs.append({"pair": [i, j], "spearman": float(rho), "n_valid": int(mask.sum())})
        stability["per_layer"][str(lid)] = pairs

    return {"layers": layers_out, "stability_spearman": stability, "moe_layer_indices": moe_layer_indices}


def compute_layer_bridge_summary(
    finalized: Dict[str, Any],
    tau_disp: float,
    eps: float,
) -> Dict[str, Any]:
    """
    由 pooled_all 计算每层 bar_B_l、sigma_l、B_z(e,l)，并标记 sigma_l_ge_tau_disp。
    供 admissibility_merge.py 读取。
    """
    moe_ids = finalized.get("moe_layer_indices") or []
    layers_summary: Dict[str, Any] = {}
    for lid in moe_ids:
        block = finalized["layers"][str(lid)]
        pa = block["pooled_all"]
        B = np.array(pa["B"], dtype=np.float64)
        M = np.array(pa["M"], dtype=np.float64)
        valid = np.isfinite(B)
        if valid.sum() == 0:
            bar_B = float("nan")
            sigma_l = float("nan")
            B_z = [float("nan")] * len(B)
        else:
            Bv = B[valid]
            bar_B = float(Bv.mean())
            sigma_l = float(Bv.std(ddof=0))
            B_z_arr = (B - bar_B) / (sigma_l + eps)
            B_z = [float(x) if np.isfinite(x) else float("nan") for x in B_z_arr]
        layers_summary[str(lid)] = {
            "layer_idx": int(lid),
            "bar_B_l": bar_B,
            "sigma_l": sigma_l,
            "sigma_l_ge_tau_disp": bool(np.isfinite(sigma_l) and sigma_l >= tau_disp),
            "n_experts": len(B),
            "B": pa["B"],
            "M": pa["M"],
            "B_z": B_z,
        }

    # 层敏感度 S_l：|bar_B_l| 在所有 MoE 层上的分位（0~100 归一为 0~1），供 Layer-First 合并预算
    abs_list: List[float] = []
    for lid in moe_ids:
        b = layers_summary[str(lid)]["bar_B_l"]
        if np.isfinite(b):
            abs_list.append(abs(float(b)))
    for lid in moe_ids:
        k = str(lid)
        b = layers_summary[k]["bar_B_l"]
        if len(abs_list) == 0 or not np.isfinite(b):
            layers_summary[k]["S_l_sensitivity"] = float("nan")
        else:
            layers_summary[k]["S_l_sensitivity"] = float(
                percentileofscore(abs_list, abs(float(b)), kind="rank") / 100.0
            )

    return {
        "meta": {
            "tau_disp": tau_disp,
            "eps": eps,
            "description": "bar_B_l、sigma_l 来自 pooled_all 的 B；B_z 为层内 z-score；S_l_sensitivity 为 |bar_B_l| 的层间分位",
        },
        "moe_layer_indices": [int(x) for x in moe_ids],
        "layers": layers_summary,
    }


def plot_distributions(
    finalized: Dict[str, Any],
    out_dir: Path,
    n_experts: int,
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    paths: List[str] = []
    layers = finalized["layers"]
    sns.set_theme(style="whitegrid")

    for lid_str, block in layers.items():
        B_all = np.array(block["pooled_all"]["B"], dtype=np.float64)
        valid = np.isfinite(B_all)
        if valid.sum() == 0:
            continue
        plt.figure(figsize=(8, 4))
        sns.histplot(B_all[valid], kde=True, bins=min(32, max(8, n_experts // 4)))
        plt.title(f"Bridge score B distribution (layer {lid_str}, pooled)")
        plt.xlabel("B")
        fn = out_dir / f"bridge_B_hist_layer_{lid_str}.png"
        plt.tight_layout()
        plt.savefig(fn, dpi=150)
        plt.close()
        paths.append(str(fn))

    # 热力图：MoE 层 x expert 的 B（pooled）
    moe_ids = finalized["moe_layer_indices"]
    if moe_ids:
        mat = []
        for lid in moe_ids:
            B_all = np.array(finalized["layers"][str(lid)]["pooled_all"]["B"], dtype=np.float64)
            mat.append(B_all)
        arr = np.stack(mat, axis=0)
        plt.figure(figsize=(max(10, n_experts // 8), max(4, len(moe_ids) * 0.35)))
        sns.heatmap(arr, cmap="viridis", xticklabels=4, yticklabels=[str(x) for x in moe_ids])
        plt.xlabel("expert id")
        plt.ylabel("layer idx")
        plt.title("Bridge score B (pooled all subsets)")
        fn = out_dir / "bridge_B_heatmap_layers_x_experts.png"
        plt.tight_layout()
        plt.savefig(fn, dpi=150)
        plt.close()
        paths.append(str(fn))

    return paths


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.layer_summary_only:
        if not args.bridge_results_json:
            raise SystemExit("--layer-summary-only 需要 --bridge-results-json 指向已有 bridge_score_results.json")
        src = Path(args.bridge_results_json).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        with open(src, "r", encoding="utf-8") as f:
            finalized = json.load(f)
        layer_summary = compute_layer_bridge_summary(
            finalized, tau_disp=args.tau_disp, eps=args.layer_summary_eps
        )
        out_path = src.parent / "layer_bridge_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(layer_summary, f, ensure_ascii=False, indent=2)
        logger.info("已从 %s 写出层汇总（离线）: %s", src, out_path)
        return

    cal_dir = Path(args.calibration_dir).expanduser().resolve()
    manifest_path = cal_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 {manifest_path}")

    results_root = Path(args.results_dir).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("bridge_score_%Y%m%d_%H%M%S")
    out_dir = results_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    if args.max_samples > 0:
        manifest = manifest[: args.max_samples]
    n_samples_total = len(manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("未检测到 CUDA，速度会极慢")

    logger.info("加载模型 %s ...", args.model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    vl_gpt = vl_gpt.to(device).eval()
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(args.model_path)

    lang = vl_gpt.language.model
    handles, captures = register_moe_hooks(lang)
    if not captures:
        raise RuntimeError("未发现 DeepseekV2MoE 模块，请检查模型是否为 MoE 版本")

    moe_layers = sorted(captures.keys())
    # 任取一层 MoE 读取 expert 数
    sample_moe = None
    for name, mod in lang.named_modules():
        if isinstance(mod, DeepseekV2MoE):
            sample_moe = mod
            break
    assert sample_moe is not None
    n_experts = int(sample_moe.config.n_routed_experts)
    logger.info("MoE 层索引: %s，每 token top-K=%s，n_routed_experts=%d", moe_layers, sample_moe.num_experts_per_tok, n_experts)

    stats: Dict[int, Dict[str, Any]] = {lid: _empty_layer_stats(n_experts, args.num_subsets) for lid in moe_layers}

    bs = max(1, args.batch_size)

    def prepare_batch_multi(
        batch_rows: List[Tuple[int, Dict[str, Any]]],
    ) -> Tuple[Any, Any, Any, Any, List[int]]:
        plist_o, plist_va, plist_mm, plist_ta = [], [], [], []
        subset_ids = []
        for global_i, rec in batch_rows:
            q = rec["question"]
            paths = rec["paths"]
            w, h = rec["image_size"]
            subset_ids.append(subset_id(global_i, n_samples_total, args.num_subsets))
            q_ta = text_ablation_question(str(q), args.text_ablation_string, not args.no_match_question_chars)

            def one(image_path: str, question: str, resize_to: Optional[Tuple[int, int]] = None):
                pil = Image.open(image_path).convert("RGB")
                if resize_to is not None and pil.size != resize_to:
                    pil = pil.resize(resize_to, Image.Resampling.BICUBIC)
                conv = [
                    {"role": "<|User|>", "content": f"<image>\n{question}", "images": [image_path]},
                    {"role": "<|Assistant|>", "content": ""},
                ]
                return processor(
                    conversations=conv,
                    images=[pil],
                    force_batchify=False,
                    system_prompt=args.system_prompt,
                )

            plist_o.append(one(paths["original"], q))
            plist_va.append(one(paths["visual_ablated"], q))
            plist_mm.append(one(paths["mismatch"], q, resize_to=(w, h)))
            plist_ta.append(one(paths["original"], q_ta))

        return (
            processor.batchify(plist_o),
            processor.batchify(plist_va),
            processor.batchify(plist_mm),
            processor.batchify(plist_ta),
            subset_ids,
        )

    idx = 0
    pbar = tqdm(total=len(manifest), desc="bridge_score", unit="ex")
    while idx < len(manifest):
        cur_bs = bs
        success = False
        idx_start = idx
        while cur_bs >= 1 and not success:
            end = min(idx + cur_bs, len(manifest))
            chunk = [(idx + k, manifest[idx + k]) for k in range(end - idx)]
            try:
                prep_o, prep_va, prep_mm, prep_ta, subset_ids = prepare_batch_multi(chunk)
                for prep in (prep_o, prep_va, prep_mm, prep_ta):
                    prep.to(device)

                clear_captures(captures)
                with torch.no_grad():
                    run_language_core(vl_gpt, prep_o, device)
                orig_snap = {
                    lid: {
                        "topk": captures[lid]["topk"].cpu() if captures[lid]["topk"] is not None else None,
                        "moe_out": captures[lid]["moe_out"].cpu() if captures[lid]["moe_out"] is not None else None,
                    }
                    for lid in moe_layers
                }

                for key, prep_alt in [("vis", prep_va), ("txt", prep_ta), ("mm", prep_mm)]:
                    clear_captures(captures)
                    with torch.no_grad():
                        run_language_core(vl_gpt, prep_alt, device)
                    for lid in moe_layers:
                        mo_alt = captures[lid]["moe_out"]
                        if mo_alt is None:
                            continue
                        mo_alt = mo_alt.cpu()
                        mo_o = orig_snap[lid]["moe_out"]
                        tk = orig_snap[lid]["topk"]
                        if mo_o is None or tk is None:
                            continue
                        accumulate_layer_batch(
                            layer_idx=lid,
                            st=stats[lid],
                            topk_cpu=tk,
                            moe_orig=mo_o,
                            moe_alt=mo_alt,
                            attention_mask_o=prep_o.attention_mask.cpu(),
                            images_seq_mask_o=prep_o.images_seq_mask.cpu(),
                            attention_mask_a=prep_alt.attention_mask.cpu(),
                            images_seq_mask_a=prep_alt.images_seq_mask.cpu(),
                            n_experts=n_experts,
                            subset_ids=subset_ids,
                            key=key,
                            seq_align=args.seq_align,
                        )

                del prep_o, prep_va, prep_mm, prep_ta
                gc.collect()
                torch.cuda.empty_cache()
                success = True
                idx = end
                pbar.update(end - idx_start)
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM at batch_size=%d，减半重试", cur_bs)
                gc.collect()
                torch.cuda.empty_cache()
                cur_bs //= 2
                if cur_bs < 1:
                    raise
        if not success:
            raise RuntimeError("无法完成 forward")

    pbar.close()

    for h in handles:
        h.remove()

    finalized = finalize_scores(stats, n_experts, args.num_subsets, args.affinity_eps)
    if args.skip_plots:
        plot_paths = []
    else:
        try:
            plot_paths = plot_distributions(finalized, out_dir, n_experts)
        except ImportError as e:
            logger.warning("跳过绘图（ImportError: %s）", e)
            plot_paths = []
    finalized["figures"] = plot_paths
    finalized["meta"] = {
        "model_path": args.model_path,
        "calibration_dir": str(cal_dir),
        "n_samples": len(manifest),
        "seed": args.seed,
        "batch_size_requested": args.batch_size,
        "text_ablation_string": args.text_ablation_string,
        "match_question_chars": not args.no_match_question_chars,
        "affinity_eps": args.affinity_eps,
        "num_subsets": args.num_subsets,
        "seq_align": args.seq_align,
        "note": "聚合位置：(attn_o & img_o & attn_alt & img_alt)；MoE 输出为 DeepseekV2MoE.forward 返回值（含 shared）；长度不一致时见 seq_align",
    }

    json_path = out_dir / "bridge_score_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(finalized, f, ensure_ascii=False, indent=2)

    layer_summary = compute_layer_bridge_summary(
        finalized, tau_disp=args.tau_disp, eps=args.layer_summary_eps
    )
    summary_path = out_dir / "layer_bridge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(layer_summary, f, ensure_ascii=False, indent=2)

    logger.info("已写入 %s，图 %s", json_path, plot_paths)
    logger.info("层汇总表（Step 3.9）: %s", summary_path)


if __name__ == "__main__":
    main()
