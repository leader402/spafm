"""Stage 5 知识增强测试。"""

from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch
from spafm.data.loaders._common import make_synthetic
from torch.utils.data import DataLoader

from spafm.knowledge import (
    GenePriorBank,
    PriorAligner,
    alignment_loss,
    build_synthetic_prior,
)
from spafm.models import ModelConfig
from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig
from spafm.training import H5ADCorpusDataset, PretrainConfig, SpaFMPretrainModule
from spafm.training.collator import make_collator


# --------------------------------------------------------------------------- #
def _vocab(symbols: list[str]) -> GeneVocab:
    return GeneVocab.from_symbols(symbols)


def test_bank_npz_roundtrip(tmp_path):
    syms = ["GENE1", "GENE2", "GENE3"]
    bank = build_synthetic_prior(syms, dim=8, seed=0)
    path = tmp_path / "p.npz"
    bank.save_npz(path)

    loaded = GenePriorBank.from_npz(path)
    assert loaded.symbols == syms
    assert loaded.dim == 8
    np.testing.assert_allclose(loaded.vectors, bank.vectors)


def test_align_to_vocab_mask():
    syms = ["GENE1", "GENE2", "GENE3"]
    bank = build_synthetic_prior(syms, dim=8)
    vocab = _vocab(["GENE1", "GENE3", "GENE99"])  # GENE99 不在 prior
    mat, mask = bank.align_to_vocab(vocab)
    assert mat.shape == (len(vocab), 8)
    # special tokens 与 GENE99 应 mask=False
    sym_to_id = vocab.symbol_to_id
    assert bool(mask[sym_to_id["GENE1"]]) is True
    assert bool(mask[sym_to_id["GENE3"]]) is True
    assert bool(mask[sym_to_id["GENE99"]]) is False
    # mask=False 行应为 0
    assert torch.all(mat[~mask] == 0)


def test_alignment_loss_zero_when_perfect():
    """E·W = P 时 cosine=1, loss=0。"""
    V, d_model, d_p = 5, 4, 4
    P = torch.randn(V, d_p)
    P = torch.nn.functional.normalize(P, dim=-1)
    mask = torch.tensor([True, True, False, True, True])
    # 让 E 与 P 完全相同；projection 用单位阵
    aligner = PriorAligner(d_model=d_model, d_prior=d_p, prior_matrix=P, prior_mask=mask)
    with torch.no_grad():
        aligner.proj.weight.copy_(torch.eye(d_p, d_model))
    loss = aligner(P)
    assert float(loss) < 1e-5


def test_alignment_loss_decreases_when_optimized():
    V, d_model, d_p = 16, 8, 12
    P = torch.randn(V, d_p)
    P = torch.nn.functional.normalize(P, dim=-1)
    mask = torch.ones(V, dtype=torch.bool)
    aligner = PriorAligner(d_model=d_model, d_prior=d_p, prior_matrix=P, prior_mask=mask)

    E = torch.nn.Parameter(torch.randn(V, d_model))
    optimizer = torch.optim.Adam([E, *aligner.parameters()], lr=0.1)
    losses = []
    for _ in range(40):
        loss = aligner(E)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.5


def test_alignment_loss_function_no_projection():
    V, d = 4, 6
    E = torch.randn(V, d)
    P = torch.randn(V, d)
    mask = torch.tensor([True, False, True, True])
    loss = alignment_loss(E, P, mask, projection=None)
    assert torch.isfinite(loss) and loss >= 0


def test_pretrain_module_with_aligner(tmp_path):
    a = make_synthetic(n_obs=8, n_vars=12, seed=0)
    f = tmp_path / "demo.h5ad"
    a.write_h5ad(f, compression="gzip")

    syms = a.var["gene_symbol"].astype(str).tolist()
    vocab = GeneVocab.from_symbols(syms)
    tcfg = TokenizerConfig(
        max_genes=12,
        expression={"mode": "bin", "n_bins": 11},
        spatial={"mode": "sincos", "dim": 16, "coord_scale": 100.0},
    )
    tok = STTokenizer(vocab=vocab, cfg=tcfg)
    ds = H5ADCorpusDataset(files=[f], tokenizer=tok)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=make_collator(tok))

    pcfg = PretrainConfig(
        model_config=ModelConfig(
            vocab_size=len(vocab),
            n_value_bins=11,
            expression_mode="bin",
            d_model=16,
            d_pos=16,
            n_layers=1,
            n_heads=2,
            d_ffn=32,
            dropout=0.0,
        ),
        knowledge={
            "enabled": True,
            "prior_path": None,
            "prior_format": "npz",
            "alignment_weight": 0.5,
            "freeze_prior": True,
        },
        optim={"lr": 1e-3, "weight_decay": 0.0, "warmup_steps": 0, "max_steps": 2},
    )
    module = SpaFMPretrainModule(pcfg)

    # 构造 aligner 并挂载
    bank = build_synthetic_prior(syms, dim=8, seed=1)
    P, M = bank.align_to_vocab(vocab)
    aligner = PriorAligner(d_model=16, d_prior=8, prior_matrix=P, prior_mask=M)
    module.attach_prior_aligner(aligner)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path / "runs"),
    )
    trainer.fit(module, loader)
    assert trainer.global_step >= 2

    # 跑一遍前向，确认 loss_align > 0 被记录
    batch = next(iter(loader))
    out = module._compute_losses(batch)
    assert "loss_align" in out
    assert float(out["loss_align"]) > 0
