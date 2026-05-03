"""空间可变基因（SVG）下游评测：基于 Hierarchical SpaFM 的内层注意力。

背景：
    SVG 是 ST 数据分析的核心任务之一。Gold standard 是 Moran's I 等空间自相关
    指标。本模块测试一个新假设：

        **若一个基因在跨多个 spot 中持续被同 spot 内其它基因强烈关注（高
        in-attention），则它倾向于是该 niche 的核心基因；这种"被关注度"
        的空间结构性 → 反映 SVG。**

打分流程（无监督）：
    1. 前向 ``HierarchicalSpaFM(..., return_inner_attn=True)`` 拿到内层
       attention：list of ``(B, N_spots, H, L, L)``。
    2. 对每个 spot 的每个基因 token i，计算 in-attention 得分
       ``s_i = mean_{j != i} A[j, i]``（i 被多少其他 token 关注）。
    3. 把 ``s_{spot, gene}`` 还原回 ``(N_spots, V_genes)`` 矩阵；
       每基因得到 N_spots 维向量，作为该基因的"注意力空间画像"。
    4. **Moran's I on attention picture**：用 spot 距离 KNN 图，对每个
       基因计算其 attention 画像的 Moran's I，得 ``I_attn[g]``。
    5. **Gold-standard**：用原始表达 log1p(X) 计算每基因的
       ``I_expr[g]``（标准 Moran's I）。
    6. **指标**：
        - ``spearman(I_attn, I_expr)`` —— 注意力发现 SVG 的能力
        - ``top_k_overlap`` —— 两种排名前 K 的 Jaccard 重叠

baselines（同样在表达层算）：
    - ``mean_expr``：基因平均表达，按高低排（低 baseline，应近 0）
    - ``var_expr``：基因方差排序（也是简单 baseline）
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
from sklearn.neighbors import NearestNeighbors

from spafm.benchmarks.ccc import _row, _safe_corr  # 复用
from spafm.models.hierarchical import HierarchicalSpaFM
from spafm.tokenization import STTokenizer
from spafm.training.slice_dataset import make_slice_collator


# --------------------------------------------------------------------------- #
@dataclass
class SVGResult:
    """SVG 评测结果。"""

    n_spots: int
    n_genes_scored: int
    moran_attn: np.ndarray  # (G,)
    moran_expr: np.ndarray  # (G,)
    gene_symbols: list[str]
    spearman_attn_vs_expr: float
    top_k_overlap: dict[int, float]  # K → jaccard
    baseline_spearman: dict[str, float] = field(default_factory=dict)
    baseline_top_k_overlap: dict[str, dict[int, float]] = field(default_factory=dict)
    top_genes_attn: list[str] = field(default_factory=list)
    top_genes_expr: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_spots": int(self.n_spots),
            "n_genes_scored": int(self.n_genes_scored),
            "spearman_attn_vs_expr": float(self.spearman_attn_vs_expr),
            "top_k_overlap": {int(k): float(v) for k, v in self.top_k_overlap.items()},
            "baseline_spearman": {k: float(v) for k, v in self.baseline_spearman.items()},
            "baseline_top_k_overlap": {
                k: {int(kk): float(vv) for kk, vv in d.items()}
                for k, d in self.baseline_top_k_overlap.items()
            },
            "top_genes_attn": list(self.top_genes_attn),
            "top_genes_expr": list(self.top_genes_expr),
        }


# --------------------------------------------------------------------------- #
# Moran's I（KNN 图，行归一化权重）
# --------------------------------------------------------------------------- #
def knn_spatial_weights(coords: np.ndarray, k: int = 8) -> sp.csr_matrix:
    """返回行归一化的 KNN 邻接矩阵（自身排除）。"""
    n = coords.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, ind = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), k)
    cols = ind[:, 1:].ravel()  # 去掉自身
    data = np.ones_like(cols, dtype=np.float64)
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # 行归一化
    deg = np.asarray(W.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    Dinv = sp.diags(1.0 / deg)
    return (Dinv @ W).tocsr()


def morans_I_batch(X: np.ndarray, W: sp.csr_matrix) -> np.ndarray:
    """对每列计算 Moran's I。

    Args:
        X: (N, G) 矩阵（每列一个基因 / 特征）。
        W: (N, N) 行归一化空间权重。
    Returns:
        I: (G,) 数组。
    """
    N, G = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    var = (Xc**2).sum(axis=0)  # (G,)
    var[var < 1e-12] = np.nan  # 常数列 → I = NaN
    # 分子：sum_{ij} W[i,j] * Xc[i] * Xc[j] = sum_i Xc[i] * (W Xc)[i]
    WXc = W @ Xc  # (N, G)
    num = (Xc * WXc).sum(axis=0)
    # 分母 (S0 / N) * var；W 已经行归一化，S0 = N（行和为 1）→ S0/N = 1
    morans_i = num / var
    return morans_i


# --------------------------------------------------------------------------- #
# 内层 attention → (N_spots, V_genes) 画像
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_inner_attention_picture(
    model: HierarchicalSpaFM,
    adata: AnnData,
    tokenizer: STTokenizer,
    *,
    device: str = "cpu",
    max_spots: int | None = 200,
    seed: int = 0,
    chunk_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """对一个 slice 跑前向，返回：

    Returns
    -------
    (picture, sel, genes_in_vocab)
        picture: shape ``(N_spots, V_picture)``，列对应 ``genes_in_vocab``，
                 元素 = 该基因在该 spot 的 in-attention 平均
                 （只在该基因实际出现在该 spot 时记分，否则 NaN）。
        sel: (N_spots,) 选中的 spot 全局索引
        genes_in_vocab: list[str] 长度 V_picture，列名（gene symbol）

    Notes
    -----
    ``chunk_size`` 控制一次前向多少 spot；inner attention 张量形状为
    ``(n_layers, 1, N, H, L, L)``，N=300 / H=8 / L=256 时单层 ~600MB，
    24GB 卡爆显存。分块后每次只缓存 ``chunk_size`` 个 spot 的 attn，
    且立刻折叠到 ``in_score (chunk, L)`` 后释放。
    """
    n = adata.n_obs
    rng = np.random.default_rng(seed)
    if max_spots is not None and n > max_spots:
        sel = np.sort(rng.choice(n, size=max_spots, replace=False))
    else:
        sel = np.arange(n)

    var_token_ids = tokenizer._gene_id_array(adata)
    coords_full = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]

    spot_dicts = []
    for ri in sel:
        row = _row(adata, int(ri))
        spot_dicts.append(
            tokenizer.encode_one(
                row_counts=row, coord=coords_full[int(ri)], var_token_ids=var_token_ids
            )
        )
    N = len(sel)
    chunk_size = max(1, int(chunk_size))

    # ---- 分块前向，累积 in_score / gene_ids / attn_mask ---- #
    gene_ids_chunks: list[np.ndarray] = []
    attn_mask_chunks: list[np.ndarray] = []
    in_score_chunks: list[np.ndarray] = []

    for s_start in range(0, N, chunk_size):
        s_end = min(N, s_start + chunk_size)
        n_chunk = s_end - s_start
        item = {
            "spot_dicts": spot_dicts[s_start:s_end],
            "spot_coords": coords_full[sel[s_start:s_end]],
            "spot_attention_mask": np.ones(n_chunk, dtype=bool),
            "n_spots_valid": np.int64(n_chunk),
            "slice_idx": np.int64(0),
        }
        coll = make_slice_collator(tokenizer, n_spots_per_sample=n_chunk)
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
            return_inner_attn=True,
        )
        inner_attns = out["inner_attentions"]  # list of (B=1, n_chunk, H, L, L)
        # 逐层累加 mean over heads；避免 torch.stack(...) 一次性占两份显存
        L_chunk = inner_attns[0].shape[-1]
        in_score_acc = torch.zeros(n_chunk, L_chunk, device=device, dtype=torch.float32)
        for a in inner_attns:
            # a: (1, n_chunk, H, L, L)；先对 head 取平均，再对行(=被多少 token 关注)取平均
            a_mean_h = a[0].float().mean(dim=1)  # (n_chunk, L, L)
            in_score_acc += a_mean_h.mean(dim=1)  # (n_chunk, L)
            del a_mean_h
        in_score_acc /= float(len(inner_attns))

        gene_ids_chunks.append(batch["gene_ids"][0].cpu().numpy())  # (n_chunk, L_chunk)
        attn_mask_chunks.append(batch["attention_mask"][0].cpu().numpy())
        in_score_chunks.append(in_score_acc.cpu().numpy().astype(np.float64))

        del out, inner_attns, in_score_acc, batch
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # 不同 chunk 的 L 可能不同（pad 到各 chunk 内部最大），需对齐到全局 L_max
    L_max = max(g.shape[1] for g in gene_ids_chunks)

    def _pad_to(arr: np.ndarray, L_target: int, fill: int | float = 0) -> np.ndarray:
        L_cur = arr.shape[1]
        if L_cur == L_target:
            return arr
        pad = np.full((arr.shape[0], L_target - L_cur), fill, dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=1)

    gene_ids_np = np.concatenate(
        [_pad_to(g, L_max, 0) for g in gene_ids_chunks], axis=0
    )  # (N, L_max)
    attn_mask_np = np.concatenate(
        [_pad_to(m, L_max, False) for m in attn_mask_chunks], axis=0
    )
    in_score_np = np.concatenate(
        [_pad_to(s, L_max, 0.0) for s in in_score_chunks], axis=0
    )

    # 收集出现过的所有 gene token id（vocab 内）
    valid_ids = gene_ids_np[attn_mask_np]
    uniq_ids, _ = np.unique(valid_ids, return_counts=True)
    # 仅取在 vocab 中的（>= 8 是 special token 阈值）
    SPECIAL = 8
    uniq_ids = uniq_ids[uniq_ids >= SPECIAL]

    # 构建 picture (N, G)；G = uniq genes 数量
    G = len(uniq_ids)
    id_to_col = {int(g): k for k, g in enumerate(uniq_ids)}
    picture = np.full((N, G), np.nan, dtype=np.float64)
    for s in range(N):
        for li in range(gene_ids_np.shape[1]):
            if not attn_mask_np[s, li]:
                continue
            gid = int(gene_ids_np[s, li])
            if gid < SPECIAL:
                continue
            col = id_to_col[gid]
            # 若同 spot 同基因多次出现（一般不会），取 mean
            cur = picture[s, col]
            v = in_score_np[s, li]
            picture[s, col] = v if np.isnan(cur) else 0.5 * (cur + v)

    # gene_id → symbol（vocab.id_to_symbol 是个 list）
    id_to_sym = tokenizer.vocab.id_to_symbol
    genes_in_vocab: list[str] = []
    for g in uniq_ids:
        gi = int(g)
        sym = id_to_sym[gi] if 0 <= gi < len(id_to_sym) else f"VOCAB_{gi}"
        genes_in_vocab.append(sym)

    return picture, sel, genes_in_vocab


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """两个排名前 K 的 jaccard 系数（按 score 降序）。"""
    finite_a = np.where(np.isfinite(a))[0]
    finite_b = np.where(np.isfinite(b))[0]
    if len(finite_a) < k or len(finite_b) < k:
        k = min(k, len(finite_a), len(finite_b))
    if k <= 0:
        return float("nan")
    top_a = set(finite_a[np.argsort(-a[finite_a])[:k]])
    top_b = set(finite_b[np.argsort(-b[finite_b])[:k]])
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return inter / union if union > 0 else float("nan")


def run_svg_analysis(
    h5ad: str | Path,
    *,
    model: HierarchicalSpaFM,
    tokenizer: STTokenizer,
    max_spots: int = 300,
    knn: int = 8,
    top_ks: Sequence[int] = (20, 50, 100),
    device: str = "cpu",
    seed: int = 0,
    min_nonnan_frac: float = 0.3,
    chunk_size: int = 32,
) -> SVGResult:
    """SVG 评测主流程。

    Args:
        min_nonnan_frac: 一个基因在 attention picture 中至少要在
            ``min_nonnan_frac * N_spots`` 个 spot 出现才参与打分。
        chunk_size: 每次前向多少 spot（控制显存峰值）。
    """
    adata = read_h5ad(h5ad)

    picture, sel, genes_in_vocab = extract_inner_attention_picture(
        model,
        adata,
        tokenizer,
        device=device,
        max_spots=max_spots,
        seed=seed,
        chunk_size=chunk_size,
    )
    N = picture.shape[0]
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)[sel, :2]
    W = knn_spatial_weights(coords, k=knn)

    # 过滤覆盖率太低的基因
    occ = (~np.isnan(picture)).mean(axis=0)  # (G,)
    keep = occ >= float(min_nonnan_frac)
    picture = picture[:, keep]
    genes_in_vocab = [g for g, k in zip(genes_in_vocab, keep, strict=True) if k]

    # NaN 用列均值填充（保留空间结构）
    col_mean = np.nanmean(picture, axis=0)
    inds = np.where(np.isnan(picture))
    picture[inds] = np.take(col_mean, inds[1])

    moran_attn = morans_I_batch(picture, W)  # (G,)

    # Gold-standard：log1p(X) 上算 Moran's I（仅取同样的基因子集）
    # 注意 vocab 的 symbol 全大写，adata 可能 mixed case，统一为 upper 匹配
    syms_full = (
        adata.var["gene_symbol"].astype(str).tolist()
        if "gene_symbol" in adata.var
        else list(adata.var_names)
    )
    sym_to_col = {s.upper(): i for i, s in enumerate(syms_full)}
    genes_upper = [g.upper() for g in genes_in_vocab]
    cols_in_X = [sym_to_col[g] for g in genes_upper if g in sym_to_col]
    valid_mask = np.array([g in sym_to_col for g in genes_upper])

    Xs = adata.X[sel]
    Xs = Xs.toarray() if sp.issparse(Xs) else np.asarray(Xs)
    Xs = np.log1p(Xs.astype(np.float32))
    X_sub = Xs[:, cols_in_X]
    moran_expr_sub = morans_I_batch(X_sub.astype(np.float64), W)

    # 把 sub 对齐回完整 G
    moran_expr = np.full_like(moran_attn, np.nan)
    moran_expr[valid_mask] = moran_expr_sub

    # 指标
    spearman_main = _safe_corr(moran_attn, moran_expr, "spearman")
    top_k_overlap = {int(k): _topk_jaccard(moran_attn, moran_expr, int(k)) for k in top_ks}

    # baselines（在表达层另算 mean / var 排序，与 moran_expr 对照看）
    mean_expr = X_sub.mean(axis=0).astype(np.float64)
    var_expr = X_sub.var(axis=0).astype(np.float64)
    baseline_spearman = {
        "mean_expr_vs_moran_expr": _safe_corr(mean_expr, moran_expr_sub, "spearman"),
        "var_expr_vs_moran_expr": _safe_corr(var_expr, moran_expr_sub, "spearman"),
    }
    baseline_top_k = {
        "mean_expr": {
            int(k): _topk_jaccard(mean_expr, moran_expr_sub, int(k)) for k in top_ks
        },
        "var_expr": {
            int(k): _topk_jaccard(var_expr, moran_expr_sub, int(k)) for k in top_ks
        },
    }

    # top 基因列表
    K_show = max(top_ks)
    finite_attn = np.where(np.isfinite(moran_attn))[0]
    finite_expr = np.where(np.isfinite(moran_expr))[0]
    top_attn_idx = finite_attn[np.argsort(-moran_attn[finite_attn])[:K_show]]
    top_expr_idx = finite_expr[np.argsort(-moran_expr[finite_expr])[:K_show]]
    top_genes_attn = [genes_in_vocab[i] for i in top_attn_idx]
    top_genes_expr = [genes_in_vocab[i] for i in top_expr_idx]

    return SVGResult(
        n_spots=int(N),
        n_genes_scored=int(picture.shape[1]),
        moran_attn=moran_attn,
        moran_expr=moran_expr,
        gene_symbols=genes_in_vocab,
        spearman_attn_vs_expr=spearman_main,
        top_k_overlap=top_k_overlap,
        baseline_spearman=baseline_spearman,
        baseline_top_k_overlap=baseline_top_k,
        top_genes_attn=top_genes_attn,
        top_genes_expr=top_genes_expr,
    )
