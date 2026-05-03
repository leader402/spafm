"""细胞-细胞通讯（CCC）下游评测：基于 Hierarchical SpaFM 的外层注意力。

核心思路：
1. 用 ``HierarchicalSpaFM`` 在一个 slice 上前向，拿到 ``outer_attentions``：
   list of ``(B=1, H, N_spots, N_spots)``。
2. 跨 head 跨 layer 平均 → spot-spot 通讯矩阵 ``A``。
3. 按 ``cell_type/niche`` 把 spot 聚合为 ``(K, K)`` 类型间通讯矩阵 ``M_attn``。
4. 给定 ligand-receptor 数据库 ``LR``，对每对 ``(L, R)``：
   ``M_lr[i, j] = mean_{s∈i, t∈j} expr(L, s) * expr(R, t)``，
   再对所有 LR 对求平均得 ``M_lr_total``。
5. 计算 ``M_attn`` 与 ``M_lr_total`` 的 spearman / pearson 相关；
   越正相关，说明模型自动学到的 spot-spot 注意力越能反映已知通讯先验。

为公平对比，另外两条 baseline 仅基于表达：
- ``cosine``：spot embedding 用 PCA(50)，pairwise cosine sim 当 A
- ``rbf``：spot 间欧氏距离的 RBF kernel 当 A（纯空间，不看表达）

输出：
- ``per_lr_corr``: 每对 LR 与 attention 的相关
- ``overall_corr``: 总通讯矩阵相关（attn vs LR-total）
- ``baseline_corrs``: cosine/rbf 的总相关
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from anndata import AnnData, read_h5ad
from scipy.stats import pearsonr, spearmanr

from spafm.models.hierarchical import HierarchicalConfig, HierarchicalSpaFM
from spafm.tokenization import STTokenizer
from spafm.training.slice_dataset import make_slice_collator

# --------------------------------------------------------------------------- #
# 内置 mouse 脑 L-R 子集（CellChatDB 风格挑选；可被外部覆盖）
# --------------------------------------------------------------------------- #
DEFAULT_MOUSE_BRAIN_LR: list[tuple[str, str]] = [
    ("Tgfb1", "Tgfbr1"),
    ("Vegfa", "Kdr"),
    ("Apoe", "Trem2"),
    ("Cxcl12", "Cxcr4"),
    ("Igf1", "Igf1r"),
    ("Wnt5a", "Fzd5"),
    ("Pdgfa", "Pdgfra"),
    ("Bmp4", "Bmpr1a"),
]


# --------------------------------------------------------------------------- #
@dataclass
class CCCResult:
    """CCC 评测结果。"""

    n_spots: int
    n_celltypes: int
    celltypes: list[str]
    M_attn: np.ndarray  # (K, K) — SpaFM outer attention 聚合
    M_lr: np.ndarray  # (K, K) — LR co-expression 聚合（所有 pair 平均）
    overall_corr_spearman: float
    overall_corr_pearson: float
    per_lr_spearman: dict[str, float] = field(default_factory=dict)
    baseline_corrs: dict[str, float] = field(default_factory=dict)
    lr_pairs_used: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_spots": int(self.n_spots),
            "n_celltypes": int(self.n_celltypes),
            "celltypes": list(self.celltypes),
            "overall_corr_spearman": float(self.overall_corr_spearman),
            "overall_corr_pearson": float(self.overall_corr_pearson),
            "per_lr_spearman": {k: float(v) for k, v in self.per_lr_spearman.items()},
            "baseline_corrs": {k: float(v) for k, v in self.baseline_corrs.items()},
            "lr_pairs_used": [list(p) for p in self.lr_pairs_used],
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _row(adata: AnnData, idx: int) -> np.ndarray:
    x = adata.X[idx]
    return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()


def _gene_to_col(adata: AnnData) -> dict[str, int]:
    syms = (
        adata.var["gene_symbol"].astype(str).tolist()
        if "gene_symbol" in adata.var
        else list(adata.var_names)
    )
    return {s: i for i, s in enumerate(syms)}


def _aggregate_to_celltype(
    A: np.ndarray, labels: np.ndarray, celltypes: list[str]
) -> np.ndarray:
    """把 (N, N) spot-spot 矩阵按标签聚合为 (K, K) celltype-celltype 矩阵（mean）。"""
    K = len(celltypes)
    M = np.zeros((K, K), dtype=np.float64)
    for i, ci in enumerate(celltypes):
        si = labels == ci
        if not si.any():
            continue
        for j, cj in enumerate(celltypes):
            sj = labels == cj
            if not sj.any():
                continue
            M[i, j] = float(A[np.ix_(si, sj)].mean())
    return M


def _safe_corr(a: np.ndarray, b: np.ndarray, kind: str = "spearman") -> float:
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    if kind == "spearman":
        r, _ = spearmanr(a, b)
    else:
        r, _ = pearsonr(a, b)
    return float(r)


# --------------------------------------------------------------------------- #
# attention 提取
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_outer_attention(
    model: HierarchicalSpaFM,
    adata: AnnData,
    tokenizer: STTokenizer,
    *,
    device: str = "cpu",
    max_spots: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """对一个 slice 跑前向，返回 ``(A_spot_spot[N,N], spot_indices[N])``。

    若 ``max_spots`` 给定且 < n_obs，则均匀随机子采样。
    """
    n = adata.n_obs
    rng = np.random.default_rng(seed)
    if max_spots is not None and n > max_spots:
        sel = np.sort(rng.choice(n, size=max_spots, replace=False))
    else:
        sel = np.arange(n)

    # 复用 SliceDataset 的逻辑：自己手工拼一个 (B=1, N, L) batch
    var_ids = tokenizer._gene_id_array(adata)
    coords_full = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
    spot_dicts = []
    for ri in sel:
        row = _row(adata, int(ri))
        spot_dicts.append(
            tokenizer.encode_one(
                row_counts=row, coord=coords_full[int(ri)], var_token_ids=var_ids
            )
        )
    N = len(sel)

    # 用 collator 的逻辑（B=1, N=N）
    item = {
        "spot_dicts": spot_dicts,
        "spot_coords": coords_full[sel],
        "spot_attention_mask": np.ones(N, dtype=bool),
        "n_spots_valid": np.int64(N),
        "slice_idx": np.int64(0),
    }
    coll = make_slice_collator(tokenizer, n_spots_per_sample=N)
    batch = coll([item])
    batch = {k: v.to(device) for k, v in batch.items()}

    out = model(
        gene_ids=batch["gene_ids"],
        pos_emb=batch["pos_emb"],
        attention_mask=batch["attention_mask"],
        spot_coords=batch["spot_coords"],
        spot_attention_mask=batch["spot_attention_mask"],
        coords=batch["coords"],
        value_ids=batch.get("value_ids"),
        value_floats=batch.get("value_floats"),
        return_attn=True,
    )
    attns = out["outer_attentions"]  # list of (1, H, N, N)
    # 跨 layer / head 平均
    A = torch.stack(attns, dim=0).mean(dim=(0, 2))[0]  # (N, N)
    return A.cpu().numpy().astype(np.float64), sel


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def baseline_pca_cosine(adata: AnnData, sel: np.ndarray, n_comp: int = 50) -> np.ndarray:
    from sklearn.decomposition import PCA

    X = adata.X[sel]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = np.log1p(X.astype(np.float32))
    n_comp = min(n_comp, min(X.shape) - 1)
    Z = PCA(n_components=n_comp, random_state=0).fit_transform(X)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    return (Z @ Z.T).astype(np.float64)


def baseline_rbf_distance(adata: AnnData, sel: np.ndarray, sigma: float | None = None) -> np.ndarray:
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)[sel, :2]
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = (diff**2).sum(-1)
    if sigma is None:
        sigma = float(np.sqrt(np.median(d2[d2 > 0])) + 1e-9)
    return np.exp(-d2 / (2.0 * sigma**2))


# --------------------------------------------------------------------------- #
# LR co-expression 矩阵
# --------------------------------------------------------------------------- #
def lr_coexpression_matrix(
    adata: AnnData, sel: np.ndarray, ligand: str, receptor: str
) -> np.ndarray | None:
    """返回 (N, N) 矩阵：``M[s,t] = expr(L, s) * expr(R, t)``（log1p 后）。"""
    g2c = _gene_to_col(adata)
    if ligand not in g2c or receptor not in g2c:
        return None
    li, ri = g2c[ligand], g2c[receptor]
    Xs = adata.X[sel]
    Xs = Xs.toarray() if sp.issparse(Xs) else np.asarray(Xs)
    Xs = np.log1p(Xs.astype(np.float32))
    L = Xs[:, li]
    R = Xs[:, ri]
    return np.outer(L, R).astype(np.float64)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_ccc_analysis(
    h5ad: str | Path,
    *,
    model: HierarchicalSpaFM | None,
    tokenizer: STTokenizer | None,
    label_key: str = "cell_type",
    lr_pairs: Sequence[tuple[str, str]] | None = None,
    max_spots: int = 600,
    device: str = "cpu",
    seed: int = 0,
    include_baselines: bool = True,
    min_spots_per_type: int = 5,
) -> CCCResult:
    """端到端 CCC 评测。

    Args:
        model: HierarchicalSpaFM 实例（可 None，仅跑 baseline 时 attn 部分缺省）。
        tokenizer: 与 model 配套的 tokenizer（model 非 None 时必须给）。
        max_spots: 控制 outer attention 复杂度（O(N^2)）。
    """
    adata = read_h5ad(h5ad)
    if label_key not in adata.obs:
        raise KeyError(f"obs 中缺少 {label_key!r} 标签列")

    pairs = list(lr_pairs) if lr_pairs is not None else DEFAULT_MOUSE_BRAIN_LR

    # SpaFM outer attention
    if model is not None:
        if tokenizer is None:
            raise ValueError("提供 model 时必须同时提供 tokenizer")
        A_attn, sel = extract_outer_attention(
            model, adata, tokenizer, device=device, max_spots=max_spots, seed=seed
        )
    else:
        rng = np.random.default_rng(seed)
        sel = (
            np.sort(rng.choice(adata.n_obs, size=max_spots, replace=False))
            if adata.n_obs > max_spots
            else np.arange(adata.n_obs)
        )
        A_attn = np.zeros((len(sel), len(sel)), dtype=np.float64)

    labels = adata.obs[label_key].astype(str).values[sel]
    # 去掉过小的类型
    uniq, counts = np.unique(labels, return_counts=True)
    keep_types = uniq[counts >= min_spots_per_type].tolist()
    keep_mask = np.isin(labels, keep_types)
    if keep_mask.sum() < 10 or len(keep_types) < 2:
        raise RuntimeError(
            f"有效 spot 太少（{int(keep_mask.sum())}）或类型不够（{len(keep_types)}）"
        )
    sel = sel[keep_mask]
    labels = labels[keep_mask]
    A_attn = A_attn[np.ix_(np.where(keep_mask)[0], np.where(keep_mask)[0])]
    celltypes = sorted(keep_types)

    # 聚合 attention
    M_attn = _aggregate_to_celltype(A_attn, labels, celltypes)

    # LR matrices
    used_pairs: list[tuple[str, str]] = []
    M_lr_per: list[np.ndarray] = []
    per_lr_spearman: dict[str, float] = {}
    for L, R in pairs:
        spot_mat = lr_coexpression_matrix(adata, sel, L, R)
        if spot_mat is None:
            continue
        used_pairs.append((L, R))
        Mp = _aggregate_to_celltype(spot_mat, labels, celltypes)
        M_lr_per.append(Mp)
        per_lr_spearman[f"{L}-{R}"] = _safe_corr(M_attn, Mp, "spearman")

    if not used_pairs:
        raise RuntimeError("没有任何 L-R 对在 var 中找到")

    M_lr = np.mean(np.stack(M_lr_per, axis=0), axis=0)
    overall_s = _safe_corr(M_attn, M_lr, "spearman")
    overall_p = _safe_corr(M_attn, M_lr, "pearson")

    baseline_corrs: dict[str, float] = {}
    if include_baselines:
        A_pca = baseline_pca_cosine(adata, sel)
        M_pca = _aggregate_to_celltype(A_pca, labels, celltypes)
        baseline_corrs["pca_cosine_spearman"] = _safe_corr(M_pca, M_lr, "spearman")
        A_rbf = baseline_rbf_distance(adata, sel)
        M_rbf = _aggregate_to_celltype(A_rbf, labels, celltypes)
        baseline_corrs["rbf_spatial_spearman"] = _safe_corr(M_rbf, M_lr, "spearman")

    return CCCResult(
        n_spots=int(len(sel)),
        n_celltypes=len(celltypes),
        celltypes=celltypes,
        M_attn=M_attn,
        M_lr=M_lr,
        overall_corr_spearman=overall_s,
        overall_corr_pearson=overall_p,
        per_lr_spearman=per_lr_spearman,
        baseline_corrs=baseline_corrs,
        lr_pairs_used=used_pairs,
    )


# --------------------------------------------------------------------------- #
# ckpt 装载
# --------------------------------------------------------------------------- #
def load_hier_from_ckpt(
    ckpt_path: str | Path,
    model_config: str | Path | dict | HierarchicalConfig,
    vocab_size_override: int | None = None,
    device: str = "cpu",
) -> HierarchicalSpaFM:
    if isinstance(model_config, (str, Path)):
        cfg = HierarchicalConfig.from_yaml(model_config)
    elif isinstance(model_config, dict):
        from spafm.models.spafm import ModelConfig

        inner = model_config.get("inner", {})
        rest = {k: v for k, v in model_config.items() if k != "inner"}
        cfg = HierarchicalConfig(inner=ModelConfig(**inner), **rest)
    else:
        cfg = model_config
    if vocab_size_override is not None:
        cfg.inner.vocab_size = int(vocab_size_override)
    model = HierarchicalSpaFM(cfg).to(device).eval()

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    bb = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
    if not bb:
        bb = sd
    missing, unexpected = model.load_state_dict(bb, strict=False)
    print(f"[load_hier_from_ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    return model
