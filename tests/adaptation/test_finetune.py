"""Stage 6 适配层单测：LoRA / heads / dataset / Lightning module。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from anndata import AnnData
from torch import nn

from spafm.adaptation import (
    CellTypeHead,
    FinetuneConfig,
    ImputationHead,
    LabeledH5ADDataset,
    LoRALinear,
    SpaFMFinetuneModule,
    SpatialDomainHead,
    apply_lora,
    count_trainable,
    mark_only_lora_as_trainable,
)
from spafm.adaptation.dataset import labeled_collate
from spafm.models import ModelConfig
from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig
from spafm.training.collator import make_collator


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
def test_lora_initial_output_equals_base() -> None:
    base = nn.Linear(8, 16)
    wrapped = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 8)
    with torch.no_grad():
        assert torch.allclose(wrapped(x), base(x))


def test_lora_output_changes_after_step() -> None:
    base = nn.Linear(8, 16)
    wrapped = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 8)
    y0 = wrapped(x).detach().clone()

    # 让 lora_B 非零
    with torch.no_grad():
        wrapped.lora_B.add_(torch.randn_like(wrapped.lora_B) * 0.1)
    y1 = wrapped(x).detach()
    assert not torch.allclose(y0, y1)

    # 原 base 参数应当被冻结
    assert not wrapped.base.weight.requires_grad


def test_apply_lora_replaces_qkv_and_out() -> None:
    cfg = ModelConfig(vocab_size=128, d_model=16, n_layers=2, n_heads=2, d_ffn=32)
    from spafm.models import SpaFMModel

    model = SpaFMModel(cfg)
    apply_lora(model, r=4, alpha=8, target_modules=("qkv", "out"))
    n_lora = sum(1 for n, _ in model.named_modules() if isinstance(_, LoRALinear))
    # 每层两个：qkv + out
    assert n_lora == cfg.n_layers * 2


def test_mark_only_lora_as_trainable() -> None:
    cfg = ModelConfig(vocab_size=128, d_model=16, n_layers=2, n_heads=2, d_ffn=32)
    from spafm.models import SpaFMModel

    model = SpaFMModel(cfg)
    apply_lora(model, r=4, alpha=8)
    mark_only_lora_as_trainable(model)
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            assert p.requires_grad
        else:
            assert not p.requires_grad
    assert count_trainable(model) > 0


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
def test_heads_shapes() -> None:
    d = 32
    B, L = 4, 10
    cell = torch.randn(B, d)
    tok = torch.randn(B, L, d)
    assert CellTypeHead(d, num_classes=5)(cell).shape == (B, 5)
    assert SpatialDomainHead(d, num_domains=7)(cell).shape == (B, 7)
    assert ImputationHead(d)(tok).shape == (B, L)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _make_demo_h5ad(path: Path, n_cells: int = 6, n_genes: int = 12) -> None:
    rng = np.random.default_rng(0)
    X = rng.poisson(0.5, size=(n_cells, n_genes)).astype(np.float32)
    var = {"gene_symbol": [f"GENE{i}" for i in range(n_genes)]}
    pool = np.array(["A", "B", "A", "C", "B", "A", "C", "B"])
    obs = {
        "cell_type": np.array([pool[i % len(pool)] for i in range(n_cells)]),
        "x": rng.uniform(0, 100, size=n_cells),
        "y": rng.uniform(0, 100, size=n_cells),
    }
    a = AnnData(X=X, obs=obs, var=var)
    a.obsm["spatial"] = np.stack([obs["x"], obs["y"]], axis=1).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path)


def _build_tokenizer(symbols: list[str]) -> STTokenizer:
    vocab = GeneVocab.from_symbols(symbols)
    return STTokenizer(vocab=vocab, cfg=TokenizerConfig())


def test_labeled_dataset_and_collate(tmp_path: Path) -> None:
    f = tmp_path / "demo.h5ad"
    _make_demo_h5ad(f)
    tok = _build_tokenizer([f"GENE{i}" for i in range(12)])
    ds = LabeledH5ADDataset(files=[f], tokenizer=tok, label_key="cell_type")
    assert ds.num_classes == 3
    item = ds[0]
    assert "label" in item

    base = make_collator(tok)
    batch = labeled_collate([ds[i] for i in range(len(ds))], base)
    assert "label" in batch
    assert batch["label"].shape == (len(ds),)
    assert batch["label"].dtype == torch.int64


# --------------------------------------------------------------------------- #
# Lightning module 烟测
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["linear_probe", "lora", "full"])
def test_finetune_module_smoke_classification(tmp_path: Path, strategy: str) -> None:
    f = tmp_path / "demo.h5ad"
    _make_demo_h5ad(f, n_cells=8)
    tok = _build_tokenizer([f"GENE{i}" for i in range(12)])
    ds = LabeledH5ADDataset(files=[f], tokenizer=tok, label_key="cell_type")

    mc = ModelConfig(
        vocab_size=len(tok.vocab),
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ffn=64,
    )
    cfg = FinetuneConfig(
        model_config=mc,
        adaptation={
            "strategy": strategy,
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["qkv", "out"]},
        },
        head={"type": "cell_type", "num_classes": ds.num_classes, "hidden": 16},
        optim={
            "lr": 1e-3,
            "weight_decay": 0.0,
            "betas": (0.9, 0.95),
            "warmup_steps": 0,
            "max_steps": 4,
        },
    )
    m = SpaFMFinetuneModule(cfg)
    base = make_collator(tok)
    batch = labeled_collate([ds[i] for i in range(len(ds))], base)
    out = m._loss(batch)
    assert torch.isfinite(out["loss"]).item()
    assert "acc" in out


def test_finetune_module_imputation_smoke(tmp_path: Path) -> None:
    f = tmp_path / "demo.h5ad"
    _make_demo_h5ad(f, n_cells=6)
    tok = _build_tokenizer([f"GENE{i}" for i in range(12)])
    from spafm.training import H5ADCorpusDataset

    ds = H5ADCorpusDataset(files=[f], tokenizer=tok)
    mc = ModelConfig(
        vocab_size=len(tok.vocab),
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ffn=64,
    )
    cfg = FinetuneConfig(
        model_config=mc,
        adaptation={"strategy": "full"},
        head={"type": "imputation", "num_classes": 1, "hidden": 16},
        optim={
            "lr": 1e-3,
            "weight_decay": 0.0,
            "betas": (0.9, 0.95),
            "warmup_steps": 0,
            "max_steps": 4,
        },
    )
    m = SpaFMFinetuneModule(cfg)
    base = make_collator(tok)
    batch = base([ds[i] for i in range(len(ds))])
    out = m._loss(batch)
    assert torch.isfinite(out["loss"]).item()
