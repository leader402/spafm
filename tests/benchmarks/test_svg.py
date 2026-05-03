"""SVG 评测模块测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from anndata import AnnData

from spafm.benchmarks.svg import (
    SVGResult,
    _topk_jaccard,
    extract_inner_attention_picture,
    knn_spatial_weights,
    morans_I_batch,
    run_svg_analysis,
)
from spafm.models import ModelConfig
from spafm.models.hierarchical import HierarchicalConfig, HierarchicalSpaFM
from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig


# --------------------------------------------------------------------------- #
def _make_spatial_adata(tmp_path: Path, seed: int = 0) -> Path:
    """构造一个带强空间结构的小 dataset：左右两半的基因表达不同。"""
    rng = np.random.default_rng(seed)
    n_per = 30
    n = 2 * n_per
    # 左半 spot 在 x ∈ [0,50]，右半在 [50,100]
    coords = np.zeros((n, 2), dtype=np.float32)
    coords[:n_per, 0] = rng.uniform(0, 50, size=n_per)
    coords[n_per:, 0] = rng.uniform(50, 100, size=n_per)
    coords[:, 1] = rng.uniform(0, 100, size=n)

    genes = ["LeftG1", "LeftG2", "RightG1", "RightG2"] + [f"Bg{i}" for i in range(20)]
    n_g = len(genes)
    X = rng.poisson(1.0, size=(n, n_g)).astype(np.float32)
    # 强空间 SVG：左半基因高表达
    X[:n_per, 0] += 15
    X[:n_per, 1] += 15
    X[n_per:, 2] += 15
    X[n_per:, 3] += 15

    a = AnnData(X=X, obs={"cell_type": ["a"] * n}, var={"gene_symbol": genes})
    a.obsm["spatial"] = coords
    p = tmp_path / "svg_demo.h5ad"
    a.write_h5ad(p)
    return p


def _make_tok(h5ad: Path) -> STTokenizer:
    from anndata import read_h5ad

    a = read_h5ad(h5ad)
    syms = a.var["gene_symbol"].astype(str).tolist()
    vocab = GeneVocab.from_symbols(syms)
    cfg = TokenizerConfig(
        max_genes=24,
        expression={"mode": "bin", "n_bins": 11},
        spatial={"mode": "sincos", "dim": 16, "coord_scale": 100.0},
    )
    return STTokenizer(vocab=vocab, cfg=cfg)


def _tiny_hier(vocab_size: int) -> HierarchicalSpaFM:
    cfg = HierarchicalConfig(
        inner=ModelConfig(
            vocab_size=vocab_size,
            n_value_bins=11,
            expression_mode="bin",
            d_model=24,
            d_pos=16,
            n_layers=1,
            n_heads=4,
            d_ffn=48,
            dropout=0.0,
        ),
        outer_n_layers=1,
        outer_n_heads=4,
        outer_d_ffn=48,
        outer_dropout=0.0,
    )
    return HierarchicalSpaFM(cfg).eval()


# --------------------------------------------------------------------------- #
def test_knn_spatial_weights_row_normalized() -> None:
    coords = np.random.RandomState(0).rand(20, 2)
    W = knn_spatial_weights(coords, k=4)
    assert W.shape == (20, 20)
    rs = np.asarray(W.sum(axis=1)).ravel()
    assert np.allclose(rs, 1.0)
    # 对角线应为 0（自身排除）
    assert (W.diagonal() == 0).all()


def test_morans_I_high_for_spatial_signal() -> None:
    """构造左右两簇的二值信号，Moran's I 应该明显 > 0。"""
    rng = np.random.default_rng(0)
    n = 60
    coords = np.zeros((n, 2))
    coords[: n // 2, 0] = rng.uniform(0, 1, size=n // 2)
    coords[n // 2 :, 0] = rng.uniform(2, 3, size=n // 2)
    coords[:, 1] = rng.uniform(0, 1, size=n)
    W = knn_spatial_weights(coords, k=5)

    # signal: 左半高右半低
    sig = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])[:, None]  # (n, 1)
    noise = rng.normal(size=(n, 1))
    X = np.hstack([sig, noise])  # (n, 2)
    mi = morans_I_batch(X, W)
    assert mi[0] > 0.5, f"strong spatial signal Moran I 应 >> noise，got {mi[0]}"
    assert abs(mi[1]) < 0.4, f"noise Moran I 应接近 0, got {mi[1]}"


def test_morans_I_constant_column_nan() -> None:
    coords = np.random.RandomState(0).rand(15, 2)
    W = knn_spatial_weights(coords, k=4)
    X = np.ones((15, 2))
    X[:, 1] = np.arange(15)  # 非常数
    mi = morans_I_batch(X, W)
    assert np.isnan(mi[0])  # 常数列
    assert np.isfinite(mi[1])


def test_topk_jaccard_basic() -> None:
    a = np.array([1.0, 2, 3, 4, 5])
    b = np.array([5.0, 4, 3, 2, 1])
    # top-2 of a = {3,4} (idx); top-2 of b = {0,1}; jaccard=0
    assert _topk_jaccard(a, b, 2) == 0.0
    # 同向
    assert _topk_jaccard(a, a, 3) == 1.0


def test_extract_inner_attention_picture(tmp_path: Path) -> None:
    p = _make_spatial_adata(tmp_path)
    tok = _make_tok(p)
    model = _tiny_hier(len(tok.vocab))
    from anndata import read_h5ad

    a = read_h5ad(p)
    pic, sel, genes = extract_inner_attention_picture(
        model, a, tok, max_spots=24, seed=0
    )
    assert pic.shape[0] == 24
    assert pic.shape[1] == len(genes)
    assert pic.shape[1] > 0
    # 至少有些位置非 NaN
    assert np.isfinite(pic).any()


def test_run_svg_end_to_end(tmp_path: Path) -> None:
    p = _make_spatial_adata(tmp_path)
    tok = _make_tok(p)
    model = _tiny_hier(len(tok.vocab))
    res = run_svg_analysis(
        h5ad=p, model=model, tokenizer=tok,
        max_spots=40, knn=5, top_ks=(5, 10),
        device="cpu", seed=0, min_nonnan_frac=0.0,
    )
    assert isinstance(res, SVGResult)
    assert res.n_genes_scored >= 4
    assert set(res.top_k_overlap.keys()) == {5, 10}
    assert "mean_expr_vs_moran_expr" in res.baseline_spearman
    # 序列化
    import json as _j

    _j.dumps(res.to_dict())


def test_run_svg_left_right_signal_recoverable(tmp_path: Path) -> None:
    """构造的左右两半基因 (LeftG1/LeftG2/RightG1/RightG2) 应有高 Moran I."""
    p = _make_spatial_adata(tmp_path)
    tok = _make_tok(p)
    model = _tiny_hier(len(tok.vocab))
    res = run_svg_analysis(
        h5ad=p, model=model, tokenizer=tok,
        max_spots=60, knn=6, top_ks=(4,),
        device="cpu", seed=0, min_nonnan_frac=0.0,
    )
    # gold-standard：moran_expr 的 top-4 应包含我们造的 4 个 spatial 基因
    upper_top_expr = {g.upper() for g in res.top_genes_expr[:8]}
    expected = {"LEFTG1", "LEFTG2", "RIGHTG1", "RIGHTG2"}
    overlap = upper_top_expr & expected
    assert len(overlap) >= 3, f"expr Moran I 应能恢复人工 SVG, got top={upper_top_expr}"
