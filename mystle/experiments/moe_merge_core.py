"""
HC-SMoE / Admissibility 共用的 MoE 专家合并：权重平均 + Gate 行平均。

约束：各 MoE 层共享同一 n_routed_experts 与同一组合并划分（与 DeepseekV2DecoderLayer 共用 config 一致）。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from deepseek_vl2.models.modeling_deepseek import DeepseekV2MLP, DeepseekV2MoE, MoEGate


def iter_moe_modules(language_model: nn.Module) -> List[DeepseekV2MoE]:
    """language_model: DeepseekV2ForCausalLM（即 vl_gpt.language）。"""
    from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE

    out: List[DeepseekV2MoE] = []
    for layer in language_model.model.layers:
        if isinstance(layer.mlp, DeepseekV2MoE):
            out.append(layer.mlp)
    return out


def expert_mlp_weight_vector(mlp: DeepseekV2MLP) -> np.ndarray:
    with torch.no_grad():
        parts = [
            mlp.gate_proj.weight.flatten(),
            mlp.up_proj.weight.flatten(),
            mlp.down_proj.weight.flatten(),
        ]
        v = torch.cat(parts, dim=0).float().cpu().numpy()
    return v


def aggregate_expert_vectors_across_moe_layers(moes: Sequence[DeepseekV2MoE]) -> np.ndarray:
    """形状 [E, D]，对每层同一 expert 的权重向量取平均。"""
    if not moes:
        raise ValueError("empty moes")
    E = len(moes[0].experts)
    vecs: List[np.ndarray] = []
    for e in range(E):
        acc: List[np.ndarray] = []
        for moe in moes:
            acc.append(expert_mlp_weight_vector(moe.experts[e]))
        vecs.append(np.mean(np.stack(acc, axis=0), axis=0))
    return np.stack(vecs, axis=0)


def resolve_target_n_routed(
    n_old: int,
    keep_ratio: float,
    n_group: Optional[int],
    num_experts_per_tok: int,
) -> int:
    """在 n_group 整除约束下，得到合并后的专家数（≤ n_old）。"""
    raw = max(int(round(n_old * keep_ratio)), num_experts_per_tok)
    raw = min(raw, n_old)
    if not n_group or n_group <= 0:
        return max(raw, num_experts_per_tok)
    g = int(n_group)
    t = raw - (raw % g)
    if t < num_experts_per_tok:
        t = ((num_experts_per_tok + g - 1) // g) * g
    t = min(max(t, num_experts_per_tok), n_old)
    if t % g != 0:
        t = t - (t % g)
        t = max(t, num_experts_per_tok)
    return t


def hierarchical_cluster_groups(
    X: np.ndarray,
    n_clusters: int,
    linkage_method: str = "average",
) -> List[List[int]]:
    """
    X: [E, D] 每行一个 expert 的特征。
    返回划分 groups，长度 = n_clusters（除非 E < n_clusters，则每专家单独成组再合并）。
    """
    E = X.shape[0]
    n_clusters = max(1, min(n_clusters, E))
    if n_clusters >= E:
        return [[i] for i in range(E)]

    dist = pdist(X, metric="cosine")
    Z = linkage(dist, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    gdict: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        gdict[int(lab)].append(i)
    groups = [sorted(gdict[k]) for k in sorted(gdict.keys())]
    # fcluster 可能少于 n_clusters（退化），此处不强制
    return groups


def average_mlp_state_dict(mlps: Sequence[DeepseekV2MLP]) -> Dict[str, torch.Tensor]:
    ref = mlps[0].state_dict()
    out: Dict[str, torch.Tensor] = {}
    for k in ref.keys():
        stacked = torch.stack([m.state_dict()[k].float() for m in mlps], dim=0)
        out[k] = stacked.mean(0).to(ref[k].dtype)
    return out


def merge_gate_state_dict(
    old_gate: MoEGate,
    groups: Sequence[Sequence[int]],
    new_gate: MoEGate,
) -> None:
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
                idx = list(g)
                nb[ni].copy_(ob[idx].float().mean().to(nb.dtype))


def build_merged_moe_module(
    old_moe: DeepseekV2MoE,
    groups: List[List[int]],
    lang_config: Any,
    device: torch.device,
) -> DeepseekV2MoE:
    """新建 DeepseekV2MoE（n_routed=len(groups)），从 old_moe 按 groups 做权重平均。"""
    new_moe = DeepseekV2MoE(lang_config).to(device)
    if old_moe.config.n_shared_experts is not None and new_moe.shared_experts is not None:
        new_moe.shared_experts.load_state_dict(old_moe.shared_experts.state_dict())

    for ni, g in enumerate(groups):
        mlps = [old_moe.experts[i] for i in g]
        avg_sd = average_mlp_state_dict(mlps)
        new_moe.experts[ni].load_state_dict(avg_sd)

    merge_gate_state_dict(old_moe.gate, groups, new_moe.gate)
    new_moe.eval()
    return new_moe


def apply_global_merge_partition(
    language_model: nn.Module,
    groups: List[List[int]],
    device: torch.device,
) -> None:
    """
    对语言塔中每个 DeepseekV2MoE 层应用同一划分 groups，并更新共享的 language_model.config.n_routed_experts。
    """
    cfg = language_model.config
    new_n = len(groups)
    cfg.n_routed_experts = new_n

    for layer in language_model.model.layers:
        if not isinstance(layer.mlp, DeepseekV2MoE):
            continue
        old = layer.mlp
        layer.mlp = build_merged_moe_module(old, groups, cfg, device)


def groups_to_mapping(groups: List[List[int]], n_old: int) -> List[int]:
    """old_expert_idx -> new_expert_idx。"""
    m = [-1] * n_old
    for ni, g in enumerate(groups):
        for i in g:
            m[i] = ni
    return m


def save_merge_plan(path: Path, groups: List[List[int]], meta: Dict[str, Any]) -> None:
    payload = {
        "meta": meta,
        "n_groups": len(groups),
        "groups": groups,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_merge_plan(path: Path) -> Tuple[List[List[int]], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["groups"], data.get("meta", {})


def pairwise_merge_admissible(
    i: int,
    j: int,
    layer_summary: Dict[str, Any],
    tau_disp: float,
    tau_z: Optional[float],
    delta_affinity: float,
    use_per_layer_tau_z: bool,
    bz_quantile_for_cutoff: float,
) -> bool:
    """
    检查 (i,j) 是否在所有「高离散」层上满足 B_z 与 M 约束。
    tau_z 若为 None，则每层用 B_z 的 bz_quantile_for_cutoff 分位数作 cutoff（小于等于该 cutoff 视为可合并）。
    """
    layers = layer_summary.get("layers", {})
    for _, row in layers.items():
        sigma_l = row.get("sigma_l")
        if sigma_l is None or not np.isfinite(sigma_l) or float(sigma_l) < tau_disp:
            continue
        Bz = row.get("B_z", [])
        M = row.get("M", [])
        if i >= len(Bz) or j >= len(Bz):
            return False
        bzi, bzj = Bz[i], Bz[j]
        if not np.isfinite(bzi) or not np.isfinite(bzj):
            return False
        if use_per_layer_tau_z:
            arr = np.array([x for x in Bz if np.isfinite(x)], dtype=np.float64)
            if arr.size == 0:
                return False
            cut = float(np.quantile(arr, bz_quantile_for_cutoff))
            if not (bzi <= cut and bzj <= cut):
                return False
        else:
            assert tau_z is not None
            if not (bzi < tau_z and bzj < tau_z):
                return False
        if i < len(M) and j < len(M) and np.isfinite(M[i]) and np.isfinite(M[j]):
            if abs(float(M[i]) - float(M[j])) >= delta_affinity:
                return False
    return True


def constrained_distance_matrix(
    X: np.ndarray,
    n_experts: int,
    layer_summary: Dict[str, Any],
    tau_disp: float,
    tau_z: Optional[float],
    delta_affinity: float,
    use_per_layer_tau_z: bool,
    bz_quantile_for_cutoff: float,
    forbidden_penalty: float,
) -> np.ndarray:
    """与 HC-SMoE 一致：余弦距离；不可合并对置为 forbidden_penalty。"""
    dist_cond = pdist(X, metric="cosine")
    D = squareform(dist_cond)
    np.fill_diagonal(D, 0.0)
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            ok = pairwise_merge_admissible(
                i,
                j,
                layer_summary,
                tau_disp,
                tau_z,
                delta_affinity,
                use_per_layer_tau_z,
                bz_quantile_for_cutoff,
            )
            if not ok:
                D[i, j] = D[j, i] = forbidden_penalty
    return D


def hierarchical_cluster_from_full_distance(
    D: np.ndarray,
    n_clusters: int,
    linkage_method: str = "average",
) -> List[List[int]]:
    """对对称距离矩阵做层次聚类（condensed）。"""
    n = D.shape[0]
    n_clusters = max(1, min(n_clusters, n))
    if n_clusters >= n:
        return [[i] for i in range(n)]

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    gdict: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        gdict[int(lab)].append(i)
    return [sorted(gdict[k]) for k in sorted(gdict.keys())]
