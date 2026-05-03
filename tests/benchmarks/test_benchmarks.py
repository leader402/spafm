"""Stage 7 评测层单测。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData

from spafm.benchmarks import (
    BenchmarkConfig,
    HVGMeanEmbedder,
    PCAEmbedder,
    SpaFMEmbedder,
    cluster_scores,
    linear_probe_cv,
    regression_scores,
    run_benchmark,
)
from spafm.models import ModelConfig
from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig


# --------------------------------------------------------------------------- #
# metrics 纯函数
# --------------------------------------------------------------------------- #
def test_linear_probe_cv_separable() -> None:
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 0.1, size=(20, 4)), rng.normal(5, 0.1, size=(20, 4))])
    y = np.array([0] * 20 + [1] * 20)
    m = linear_probe_cv(X, y, n_folds=5, seed=0)
    assert m["accuracy"] > 0.95 and m["macro_f1"] > 0.95


def test_cluster_scores_separable() -> None:
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 0.1, size=(20, 4)), rng.normal(5, 0.1, size=(20, 4))])
    y = np.array([0] * 20 + [1] * 20)
    m = cluster_scores(X, y)
    assert m["ari"] > 0.9 and m["nmi"] > 0.9


def test_regression_scores_perfect() -> None:
    t = np.linspace(0, 1, 50)
    m = regression_scores(t, t)
    assert m["pearson"] == pytest.approx(1.0)
    assert m["mse"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_demo_h5ad(path: Path, n_cells: int = 12, n_genes: int = 16) -> None:
    rng = np.random.default_rng(0)
    X = rng.poisson(0.7, size=(n_cells, n_genes)).astype(np.float32)
    var = {"gene_symbol": [f"GENE{i}" for i in range(n_genes)]}
    pool_ct = ["A", "B", "C"]
    pool_dom = ["d0", "d1"]
    obs = {
        "cell_type": np.array([pool_ct[i % len(pool_ct)] for i in range(n_cells)]),
        "niche_label": np.array([pool_dom[i % len(pool_dom)] for i in range(n_cells)]),
    }
    a = AnnData(X=X, obs=obs, var=var)
    coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
    a.obsm["spatial"] = coords
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path)


def _build_tokenizer(symbols: list[str]) -> STTokenizer:
    return STTokenizer(vocab=GeneVocab.from_symbols(symbols), cfg=TokenizerConfig())


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def test_pca_embedder(tmp_path: Path) -> None:
    f = tmp_path / "demo.h5ad"
    _make_demo_h5ad(f, n_cells=10, n_genes=20)
    out = PCAEmbedder(n_components=4).embed([f])
    assert out["cell_repr"].shape == (10, 4)


def test_hvg_mean_embedder(tmp_path: Path) -> None:
    f = tmp_path / "demo.h5ad"
    _make_demo_h5ad(f, n_cells=10, n_genes=20)
    out = HVGMeanEmbedder(top_k=5).embed([f])
    assert out["cell_repr"].shape == (10, 5)


# --------------------------------------------------------------------------- #
# SpaFMEmbedder
# --------------------------------------------------------------------------- #
def test_spafm_embedder_shape(tmp_path: Path) -> None:
    f = tmp_path / "demo.h5ad"
    _make_demo_h5ad(f, n_cells=8, n_genes=12)
    tok = _build_tokenizer([f"GENE{i}" for i in range(12)])
    mc = ModelConfig(vocab_size=len(tok.vocab), d_model=16, n_layers=2, n_heads=2, d_ffn=32)
    emb = SpaFMEmbedder(model_config=mc, tokenizer=tok, batch_size=4)
    out = emb.embed([f])
    assert out["cell_repr"].shape == (8, 16)
    assert len(out["token_repr"]) == 8


# --------------------------------------------------------------------------- #
# run_benchmark e2e
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("etype", ["pca", "spafm"])
def test_run_benchmark_e2e(tmp_path: Path, etype: str) -> None:
    f = tmp_path / "demo.h5ad"
    _make_demo_h5ad(f, n_cells=12, n_genes=12)

    # 临时 tokenizer config
    tok_cfg = tmp_path / "tok.yaml"
    tok_cfg.write_text(
        "vocab_path: null\nmax_genes: 32\ngene_select: top_k\n"
        "expression: {mode: bin, n_bins: 51}\n"
        "spatial: {mode: sincos, dim: 16, coord_scale: 1000.0}\n"
        "add_cls: true\nadd_niche: false\nseed: 0\n",
        encoding="utf-8",
    )

    embedder_cfg: dict = {"type": etype}
    if etype == "spafm":
        embedder_cfg.update(
            {
                "model_config": {
                    "vocab_size": 64,
                    "d_model": 16,
                    "d_pos": 16,
                    "n_layers": 2,
                    "n_heads": 2,
                    "d_ffn": 32,
                    "dropout": 0.0,
                },
                "ckpt": None,
                "batch_size": 4,
                "device": "cpu",
            }
        )

    cfg = BenchmarkConfig(
        data={"h5ad_glob": str(f), "tokenizer_config": str(tok_cfg)},
        embedder=embedder_cfg,
        tasks=[
            {"name": "cell_type", "label_key": "cell_type", "cv_folds": 3},
            {"name": "spatial_domain", "label_key": "niche_label"},
        ]
        + ([{"name": "imputation", "mask_ratio": 0.3}] if etype == "spafm" else []),
        output={"json_path": str(tmp_path / "out.json")},
        seed=0,
    )
    out = run_benchmark(cfg)
    assert out["results"]
    assert (tmp_path / "out.json").exists()
    # 至少应包含 accuracy 与 ari
    metrics = {(r["task"], r["metric"]) for r in out["results"]}
    assert ("cell_type", "accuracy") in metrics
    assert ("spatial_domain", "ari") in metrics
