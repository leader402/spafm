"""SpaFMModel 端到端测试。"""

from __future__ import annotations

import torch
from spafm.data.loaders._common import make_synthetic

from spafm.models import (
    ContrastiveHead,
    ModelConfig,
    SpaFMModel,
    batch_to_tensors,
    count_parameters,
)
from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig


def _small_model_cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=64,
        n_value_bins=11,
        expression_mode="bin",
        d_model=32,
        d_pos=16,
        n_layers=2,
        n_heads=4,
        d_ffn=64,
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _make_batch(mode: str = "bin", n_obs: int = 4, n_vars: int = 16, max_genes: int = 16):
    adata = make_synthetic(n_obs=n_obs, n_vars=n_vars, seed=0)
    symbols = adata.var["gene_symbol"].astype(str).tolist()
    vocab = GeneVocab.from_symbols(symbols)
    tok_cfg = TokenizerConfig(
        max_genes=max_genes,
        expression={"mode": mode, "n_bins": 11},
        spatial={"mode": "sincos", "dim": 16, "coord_scale": 100.0},
    )
    tok = STTokenizer(vocab=vocab, cfg=tok_cfg)
    batch_np = tok.encode(adata)
    return adata, vocab, batch_to_tensors(batch_np)


def test_forward_bin_mode_shapes():
    _, vocab, batch = _make_batch(mode="bin", max_genes=16)
    cfg = _small_model_cfg(vocab_size=len(vocab), expression_mode="bin")
    model = SpaFMModel(cfg).eval()
    with torch.no_grad():
        out = model(**batch, return_gene_logits=True)
    B, L = batch["gene_ids"].shape
    assert out["token_repr"].shape == (B, L, cfg.d_model)
    assert out["cell_repr"].shape == (B, cfg.d_model)
    assert out["gene_logits"].shape == (B, L, cfg.vocab_size)
    # 数值正常
    assert torch.isfinite(out["token_repr"]).all()


def test_forward_continuous_mode():
    _, vocab, batch = _make_batch(mode="continuous", max_genes=16)
    cfg = _small_model_cfg(vocab_size=len(vocab), expression_mode="continuous")
    model = SpaFMModel(cfg).eval()
    with torch.no_grad():
        out = model(**batch)
    assert "value_floats" in batch
    assert out["cell_repr"].shape[-1] == cfg.d_model


def test_spatial_bias_disabled():
    _, vocab, batch = _make_batch(max_genes=16)
    cfg = _small_model_cfg(vocab_size=len(vocab), spatial_bias={"enabled": False, "sigma": 1.0})
    model = SpaFMModel(cfg).eval()
    with torch.no_grad():
        out = model(**batch)
    assert out["token_repr"].shape[-1] == cfg.d_model


def test_backward_pass():
    _, vocab, batch = _make_batch(max_genes=16)
    cfg = _small_model_cfg(vocab_size=len(vocab))
    model = SpaFMModel(cfg).train()
    out = model(**batch, return_gene_logits=True)
    # 简单 MGM loss：把 value_ids 当伪 label，演示梯度可回传
    loss = torch.nn.functional.cross_entropy(
        out["gene_logits"].reshape(-1, cfg.vocab_size),
        batch["gene_ids"].reshape(-1),
        ignore_index=0,
    )
    loss.backward()
    # 至少一个参数有梯度
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_tied_weights_share_storage():
    _, vocab, _ = _make_batch(max_genes=8)
    cfg = _small_model_cfg(vocab_size=len(vocab), tie_gene_embedding=True)
    model = SpaFMModel(cfg)
    assert model.mgm_head.tied_weight.data_ptr() == model.embed.gene.emb.weight.data_ptr()


def test_attention_mask_padding_invariance():
    """对 padding token 的 value 取任意值，cell_repr 应不变。"""
    _, vocab, batch = _make_batch(max_genes=32)
    cfg = _small_model_cfg(vocab_size=len(vocab))
    model = SpaFMModel(cfg).eval()
    with torch.no_grad():
        out1 = model(**batch)
        # 改写所有 padding 位置（attention_mask=False）的 gene_ids、pos_emb、coords
        b2 = {k: v.clone() if hasattr(v, "clone") else v for k, v in batch.items()}
        invalid = ~b2["attention_mask"]
        b2["gene_ids"][invalid] = 5  # 任意非零 id
        b2["pos_emb"][invalid] = 7.5
        b2["coords"][invalid] = 999.0
        if "value_ids" in b2:
            b2["value_ids"][invalid] = 3
        out2 = model(**b2)
    torch.testing.assert_close(out1["cell_repr"], out2["cell_repr"], rtol=1e-5, atol=1e-5)


def test_param_count_reasonable():
    cfg = ModelConfig()  # 默认 spafm-s
    model = SpaFMModel(cfg)
    n = count_parameters(model)
    # 64000 vocab × 256 = 16M 词嵌入 + transformer ~ 5M ≈ 20M
    assert 5_000_000 < n < 50_000_000


def test_contrastive_head():
    head = ContrastiveHead(d_model=32, d_proj=16)
    z = head(torch.randn(4, 32))
    assert z.shape == (4, 16)
    # L2 归一化
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_yaml_config_roundtrip(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "vocab_size: 32\nn_value_bins: 5\nexpression_mode: bin\n"
        "d_model: 16\nd_pos: 8\nn_layers: 1\nn_heads: 2\nd_ffn: 32\n"
        "dropout: 0.0\n"
        "spatial_bias: {enabled: true, sigma: 100.0}\n"
        "tie_gene_embedding: true\n",
        encoding="utf-8",
    )
    model = SpaFMModel.from_config(p)
    assert model.cfg.d_model == 16
    assert model.cfg.n_layers == 1
