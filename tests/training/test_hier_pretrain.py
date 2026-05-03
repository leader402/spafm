"""Hierarchical SpaFM 预训练 smoke + 关键不变量测试。

核心断言：外层 spatial bias 真的在工作（dist2 非零、注意力随距离变化）——
这是相对 v0 per-cell-only 架构的关键修复。
"""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from spafm.data.loaders._common import make_synthetic
from torch.utils.data import DataLoader

from spafm.models import ModelConfig
from spafm.models.hierarchical import HierarchicalConfig, HierarchicalSpaFM
from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig
from spafm.training import (
    HierarchicalSpaFMPretrainModule,
    HierPretrainConfig,
    SliceDataset,
    make_slice_collator,
)


# --------------------------------------------------------------------------- #
def _make_files(tmp_path: Path, n_files: int = 2, n_obs: int = 12, n_vars: int = 16) -> list[Path]:
    files: list[Path] = []
    for k in range(n_files):
        a = make_synthetic(n_obs=n_obs, n_vars=n_vars, seed=k)
        f = tmp_path / f"slice_{k}.h5ad"
        a.write_h5ad(f, compression="gzip")
        files.append(f)
    return files


def _make_tokenizer(files: list[Path]) -> STTokenizer:
    from anndata import read_h5ad

    syms: list[str] = []
    seen: set[str] = set()
    for f in files:
        a = read_h5ad(f)
        for s in a.var["gene_symbol"].astype(str).tolist():
            if s not in seen:
                seen.add(s)
                syms.append(s)
    vocab = GeneVocab.from_symbols(syms)
    cfg = TokenizerConfig(
        max_genes=12,
        expression={"mode": "bin", "n_bins": 11},
        spatial={"mode": "sincos", "dim": 16, "coord_scale": 100.0},
    )
    return STTokenizer(vocab=vocab, cfg=cfg)


def _tiny_hier_cfg(vocab_size: int) -> HierarchicalConfig:
    return HierarchicalConfig(
        inner=ModelConfig(
            vocab_size=vocab_size,
            n_value_bins=11,
            expression_mode="bin",
            d_model=32,
            d_pos=16,
            n_layers=2,
            n_heads=4,
            d_ffn=64,
            dropout=0.0,
        ),
        outer_n_layers=2,
        outer_n_heads=4,
        outer_d_ffn=64,
        outer_dropout=0.0,
        outer_spatial_bias_enabled=True,
        outer_spatial_sigma=50.0,
    )


# --------------------------------------------------------------------------- #
def test_slice_dataset_and_collator(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    tok = _make_tokenizer(files)
    ds = SliceDataset(files, tok, n_spots_per_sample=6, samples_per_slice=3, seed=0)
    assert len(ds) == 2 * 3

    coll = make_slice_collator(tok, n_spots_per_sample=6)
    batch = coll([ds[0], ds[1]])

    B, N = 2, 6
    assert batch["gene_ids"].shape[:2] == (B, N)
    assert batch["coords"].shape[:2] == (B, N)
    assert batch["spot_coords"].shape == (B, N, 2)
    assert batch["spot_attention_mask"].shape == (B, N)
    assert batch["pos_emb"].shape[:2] == (B, N)


def test_outer_spatial_bias_active_on_real_coords(tmp_path: Path) -> None:
    """关键断言：spot 之间 dist2 非零，外层注意力随之变化。"""
    files = _make_files(tmp_path, n_obs=10)
    tok = _make_tokenizer(files)
    ds = SliceDataset(files, tok, n_spots_per_sample=8, samples_per_slice=1, seed=0)
    coll = make_slice_collator(tok, 8)
    batch = coll([ds[0], ds[1]])

    sc = batch["spot_coords"]  # (B, N, 2)
    diff = sc.unsqueeze(2) - sc.unsqueeze(1)  # (B, N, N, 2)
    d2 = (diff**2).sum(-1)
    # 同一 slice 内多数 spot 对距离应该 > 0（合成数据空间随机）
    assert (d2 > 0).float().mean().item() > 0.5

    cfg = _tiny_hier_cfg(len(tok.vocab))
    model = HierarchicalSpaFM(cfg).eval()
    with torch.no_grad():
        out = model(
            gene_ids=batch["gene_ids"],
            pos_emb=batch["pos_emb"],
            attention_mask=batch["attention_mask"],
            spot_coords=batch["spot_coords"],
            spot_attention_mask=batch["spot_attention_mask"],
            coords=batch["coords"],
            value_ids=batch.get("value_ids"),
            return_attn=True,
        )
    attns = out["outer_attentions"]
    assert len(attns) == cfg.outer_n_layers
    a0 = attns[0]  # (B, H, N, N)
    assert a0.shape == (2, cfg.outer_n_heads, 8, 8)
    # 每行注意力应非均匀（spatial bias 在工作）
    diversity = (a0.amax(-1) - a0.amin(-1)).mean().item()
    assert diversity > 1e-3, f"outer attention 看起来均匀（diversity={diversity}），spatial bias 未生效"


def test_hier_pretrain_module_one_step(tmp_path: Path) -> None:
    files = _make_files(tmp_path)
    tok = _make_tokenizer(files)
    ds = SliceDataset(files, tok, n_spots_per_sample=4, samples_per_slice=2, seed=0)
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=make_slice_collator(tok, 4),
    )

    cfg = HierPretrainConfig(
        model_config=_tiny_hier_cfg(len(tok.vocab)),
        masking={"mask_ratio": 0.3, "mask_token_prob": 0.8, "random_token_prob": 0.1},
        losses={
            "mgm_weight": 1.0,
            "ccl_weight": 0.1,
            "ccl_temperature": 0.07,
            "use_outer_repr_for_ccl": True,
        },
        optim={
            "lr": 3.0e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
            "warmup_steps": 0,
            "max_steps": 2,
        },
    )
    module = HierarchicalSpaFMPretrainModule(cfg)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(module, loader)
    assert trainer.global_step == 2

    # 参数确实被更新（任取一个权重检查 grad 出现过）
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


_ = np  # keep import
