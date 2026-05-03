"""tokenization 模块单元测试。"""

from __future__ import annotations

import numpy as np
import pytest
from spafm.data.loaders._common import make_synthetic

from spafm.tokenization import (
    SPECIAL_TOKENS,
    GeneVocab,
    STTokenizer,
    TokenizerConfig,
    bin_expression,
    rff2d,
    sincos2d,
)
from spafm.tokenization.gene_vocab import CLS_ID, PAD_ID, UNK_ID


# --------------------------------------------------------------------------- #
# GeneVocab
# --------------------------------------------------------------------------- #
def test_gene_vocab_special_tokens_first():
    vocab = GeneVocab.from_symbols(["MT-CO1", "Gapdh", "ACTB"])
    assert len(SPECIAL_TOKENS) == 8
    assert vocab.id_to_symbol[:8] == list(SPECIAL_TOKENS)
    assert vocab.n_special == 8
    assert vocab.n_genes == 3


def test_gene_vocab_encode_unk():
    vocab = GeneVocab.from_symbols(["GAPDH"])
    ids = vocab.encode(["gapdh", "FAKE_GENE"])
    assert ids[0] == vocab.symbol_to_id["GAPDH"]
    assert ids[1] == UNK_ID


def test_gene_vocab_tsv_roundtrip(tmp_path):
    import pandas as pd

    df = pd.DataFrame(
        {
            "token_id": [0, 1, 2],
            "symbol": ["GAPDH", "ACTB", "MT-CO1"],
            "species": ["human", "human", "human"],
            "source": ["test", "test", "test"],
        }
    )
    p = tmp_path / "vocab.tsv"
    df.to_csv(p, sep="\t", index=False)
    vocab = GeneVocab.from_tsv(p)
    assert vocab.n_genes == 3
    assert "GAPDH" in vocab.symbol_to_id
    assert vocab.symbol_to_id["GAPDH"] >= 8


# --------------------------------------------------------------------------- #
# expression
# --------------------------------------------------------------------------- #
def test_bin_expression_zero_stays_zero():
    bins = bin_expression(np.array([0, 0, 5, 10, 100], dtype=np.float32), n_bins=11)
    assert bins[0] == 0 and bins[1] == 0
    assert (bins[2:] >= 1).all() and (bins[2:] <= 10).all()
    # 单调：值越大 bin 越大
    assert bins[2] <= bins[3] <= bins[4]


def test_bin_expression_all_zero():
    bins = bin_expression(np.zeros(5, dtype=np.float32))
    assert (bins == 0).all()


# --------------------------------------------------------------------------- #
# spatial encoding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [64, 128, 256])
def test_sincos2d_shape_and_range(dim):
    coords = np.random.RandomState(0).rand(20, 2).astype(np.float32)
    pe = sincos2d(coords, dim=dim)
    assert pe.shape == (20, dim)
    assert np.all(pe >= -1.0001) and np.all(pe <= 1.0001)


def test_rff2d_shape():
    coords = np.random.RandomState(0).rand(10, 2).astype(np.float32)
    pe = rff2d(coords, dim=64, sigma=1.0, seed=0)
    assert pe.shape == (10, 64)
    # 同 seed → 可复现
    pe2 = rff2d(coords, dim=64, sigma=1.0, seed=0)
    np.testing.assert_allclose(pe, pe2)


def test_sincos2d_invalid_dim():
    with pytest.raises(ValueError):
        sincos2d(np.zeros((1, 2)), dim=30)  # 非 4 倍数


# --------------------------------------------------------------------------- #
# STTokenizer 端到端
# --------------------------------------------------------------------------- #
def _build_tokenizer(max_genes: int = 32, mode: str = "bin", add_niche: bool = False):
    adata = make_synthetic(n_obs=8, n_vars=20, seed=1)
    symbols = adata.var["gene_symbol"].astype(str).tolist()
    vocab = GeneVocab.from_symbols(symbols)
    cfg = TokenizerConfig(
        max_genes=max_genes,
        expression={"mode": mode, "n_bins": 11},
        spatial={"mode": "sincos", "dim": 16, "coord_scale": 100.0},
        add_niche=add_niche,
    )
    return adata, STTokenizer(vocab=vocab, cfg=cfg)


def test_tokenizer_bin_mode_shapes():
    adata, tok = _build_tokenizer(max_genes=16, mode="bin")
    batch = tok.encode(adata)
    B, L = adata.n_obs, 16
    assert batch["gene_ids"].shape == (B, L)
    assert batch["value_ids"].shape == (B, L)
    assert batch["coords"].shape == (B, L, 2)
    assert batch["pos_emb"].shape == (B, L, 16)
    assert batch["attention_mask"].shape == (B, L)
    # CLS 在每行第 0 位
    assert (batch["gene_ids"][:, 0] == CLS_ID).all()
    # padding 处 mask=False
    for b in range(B):
        n_valid = int(batch["attention_mask"][b].sum())
        assert (batch["gene_ids"][b, n_valid:] == PAD_ID).all()


def test_tokenizer_continuous_mode():
    adata, tok = _build_tokenizer(max_genes=16, mode="continuous")
    batch = tok.encode(adata)
    assert "value_floats" in batch
    assert batch["value_floats"].dtype == np.float32


def test_tokenizer_niche_token():
    adata, tok = _build_tokenizer(max_genes=64, add_niche=True)
    batch = tok.encode(adata)
    # 至少有一个 NICHE token 出现
    from spafm.tokenization.gene_vocab import NICHE_ID

    assert (batch["gene_ids"] == NICHE_ID).any()


def test_tokenizer_yaml_roundtrip(tmp_path):
    p = tmp_path / "tok.yaml"
    p.write_text(
        "vocab_path: null\n"
        "max_genes: 8\n"
        "expression: {mode: bin, n_bins: 5}\n"
        "spatial: {mode: rff, dim: 8, coord_scale: 10.0, sigma: 1.0}\n"
        "add_cls: true\n",
        encoding="utf-8",
    )
    cfg = TokenizerConfig.from_yaml(p)
    tok = STTokenizer.from_config(cfg)
    assert tok.cfg.max_genes == 8
    assert tok.cfg.spatial["mode"] == "rff"


def test_tokenizer_works_on_all_platform_demos():
    """tokenizer 必须能跑通所有 7 平台的 synthesize_demo 输出。"""
    import importlib

    loaders = [
        "spafm.data.loaders.visium",
        "spafm.data.loaders.xenium",
        "spafm.data.loaders.merfish",
        "spafm.data.loaders.stereoseq",
        "spafm.data.loaders.slideseq2",
        "spafm.data.loaders.cosmx",
        "spafm.data.loaders.starmap",
    ]
    for name in loaders:
        adata = importlib.import_module(name).synthesize_demo()
        symbols = adata.var["gene_symbol"].astype(str).tolist()
        vocab = GeneVocab.from_symbols(symbols)
        cfg = TokenizerConfig(
            max_genes=64,
            expression={"mode": "bin", "n_bins": 11},
            spatial={"mode": "sincos", "dim": 16, "coord_scale": 1000.0},
        )
        batch = STTokenizer(vocab=vocab, cfg=cfg).encode(adata)
        assert batch["gene_ids"].shape == (adata.n_obs, 64)
        assert batch["attention_mask"].any()
