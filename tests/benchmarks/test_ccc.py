"""CCC 评测模块测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData

from spafm.benchmarks.ccc import (
    CCCResult,
    _aggregate_to_celltype,
    _safe_corr,
    baseline_pca_cosine,
    baseline_rbf_distance,
    extract_outer_attention,
    lr_coexpression_matrix,
    run_ccc_analysis,
)
from spafm.models import ModelConfig
from spafm.models.hierarchical import HierarchicalConfig, HierarchicalSpaFM
from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig


# --------------------------------------------------------------------------- #
def _make_adata(tmp_path: Path, n_obs: int = 60, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    genes = ["Tgfb1", "Tgfbr1", "Vegfa", "Kdr", "Apoe", "Trem2"] + [
        f"G{i}" for i in range(20)
    ]
    n_g = len(genes)
    X = rng.poisson(0.8, size=(n_obs, n_g)).astype(np.float32)
    # 让 cluster_0 高表达 Tgfb1，cluster_1 高表达 Tgfbr1，制造一个"通讯"
    labels = np.array([f"cluster_{i % 3}" for i in range(n_obs)])
    X[labels == "cluster_0", 0] += 10  # Tgfb1
    X[labels == "cluster_1", 1] += 10  # Tgfbr1

    coords = rng.uniform(0, 500, size=(n_obs, 2)).astype(np.float32)
    a = AnnData(
        X=X,
        obs={"cell_type": labels, "niche_label": labels},
        var={"gene_symbol": genes},
    )
    a.obsm["spatial"] = coords
    p = tmp_path / "demo.h5ad"
    a.write_h5ad(p)
    return p


def _make_tokenizer(h5ad: Path) -> STTokenizer:
    from anndata import read_h5ad

    a = read_h5ad(h5ad)
    syms = a.var["gene_symbol"].astype(str).tolist()
    vocab = GeneVocab.from_symbols(syms)
    cfg = TokenizerConfig(
        max_genes=16,
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
        outer_spatial_bias_enabled=True,
        outer_spatial_sigma=200.0,
    )
    return HierarchicalSpaFM(cfg).eval()


# --------------------------------------------------------------------------- #
def test_aggregate_to_celltype_basic() -> None:
    A = np.arange(16, dtype=np.float64).reshape(4, 4)
    labels = np.array(["a", "b", "a", "b"])
    M = _aggregate_to_celltype(A, labels, ["a", "b"])
    assert M.shape == (2, 2)
    # M[a,a] = mean(A[[0,2]][:,[0,2]]) = mean(0,2,8,10) = 5
    assert abs(M[0, 0] - 5.0) < 1e-9
    assert abs(M[1, 1] - 10.0) < 1e-9  # mean(5,7,13,15)


def test_safe_corr_zero_var() -> None:
    a = np.zeros(5)
    b = np.arange(5, dtype=float)
    assert np.isnan(_safe_corr(a, b))


def test_lr_coexpression_matrix(tmp_path: Path) -> None:
    p = _make_adata(tmp_path, n_obs=20)
    from anndata import read_h5ad

    a = read_h5ad(p)
    sel = np.arange(a.n_obs)
    M = lr_coexpression_matrix(a, sel, "Tgfb1", "Tgfbr1")
    assert M is not None and M.shape == (20, 20)
    # 不存在的基因
    assert lr_coexpression_matrix(a, sel, "NoSuchGene", "Tgfbr1") is None


def test_baselines_shapes(tmp_path: Path) -> None:
    p = _make_adata(tmp_path, n_obs=24)
    from anndata import read_h5ad

    a = read_h5ad(p)
    sel = np.arange(a.n_obs)
    P = baseline_pca_cosine(a, sel, n_comp=4)
    R = baseline_rbf_distance(a, sel)
    assert P.shape == R.shape == (24, 24)
    # 对角约为 1
    assert abs(P.diagonal().mean() - 1.0) < 1e-3
    assert abs(R.diagonal().mean() - 1.0) < 1e-6


def test_extract_outer_attention(tmp_path: Path) -> None:
    p = _make_adata(tmp_path, n_obs=24)
    tok = _make_tokenizer(p)
    model = _tiny_hier(len(tok.vocab))
    from anndata import read_h5ad

    a = read_h5ad(p)
    A, sel = extract_outer_attention(model, a, tok, max_spots=12, seed=0)
    assert A.shape == (12, 12)
    assert len(sel) == 12
    # 每行 softmax 应该约和为 1
    row_sums = A.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3)


def test_run_ccc_end_to_end(tmp_path: Path) -> None:
    p = _make_adata(tmp_path, n_obs=60)
    tok = _make_tokenizer(p)
    model = _tiny_hier(len(tok.vocab))
    res = run_ccc_analysis(
        h5ad=p,
        model=model,
        tokenizer=tok,
        label_key="cell_type",
        lr_pairs=[("Tgfb1", "Tgfbr1"), ("Vegfa", "Kdr"), ("Apoe", "Trem2")],
        max_spots=40,
        device="cpu",
        seed=0,
        min_spots_per_type=3,
    )
    assert isinstance(res, CCCResult)
    assert res.n_celltypes >= 2
    assert res.M_attn.shape == res.M_lr.shape == (res.n_celltypes, res.n_celltypes)
    assert len(res.lr_pairs_used) == 3
    assert "pca_cosine_spearman" in res.baseline_corrs
    assert "rbf_spatial_spearman" in res.baseline_corrs
    # to_dict 应可序列化
    d = res.to_dict()
    import json as _j

    _j.dumps(d)


def test_run_ccc_missing_label(tmp_path: Path) -> None:
    p = _make_adata(tmp_path, n_obs=20)
    tok = _make_tokenizer(p)
    model = _tiny_hier(len(tok.vocab))
    with pytest.raises(KeyError):
        run_ccc_analysis(
            h5ad=p, model=model, tokenizer=tok, label_key="no_such",
            max_spots=10, device="cpu",
        )
