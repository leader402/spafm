"""Stage 4 训练层 smoke test。"""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from spafm.data.loaders._common import make_synthetic
from torch.utils.data import DataLoader

from spafm.models import ModelConfig
from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig
from spafm.training import (
    H5ADCorpusDataset,
    PretrainConfig,
    SpaFMPretrainModule,
    apply_mgm_mask,
    info_nce,
    mgm_loss,
)
from spafm.training.collator import make_collator


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _make_corpus(tmp_path: Path, n_files: int = 2, n_obs: int = 6, n_vars: int = 12) -> list[Path]:
    files: list[Path] = []
    for k in range(n_files):
        a = make_synthetic(n_obs=n_obs, n_vars=n_vars, seed=k)
        f = tmp_path / f"demo_{k}.h5ad"
        a.write_h5ad(f, compression="gzip")
        files.append(f)
    return files


def _make_tokenizer(files: list[Path]) -> STTokenizer:
    from anndata import read_h5ad

    syms: list[str] = []
    seen = set()
    for f in files:
        a = read_h5ad(f)
        for s in a.var["gene_symbol"].astype(str).tolist():
            if s not in seen:
                seen.add(s)
                syms.append(s)
    vocab = GeneVocab.from_symbols(syms)
    cfg = TokenizerConfig(
        max_genes=16,
        expression={"mode": "bin", "n_bins": 11},
        spatial={"mode": "sincos", "dim": 16, "coord_scale": 100.0},
    )
    return STTokenizer(vocab=vocab, cfg=cfg)


def _tiny_model_cfg(vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        n_value_bins=11,
        expression_mode="bin",
        d_model=32,
        d_pos=16,
        n_layers=2,
        n_heads=4,
        d_ffn=64,
        dropout=0.0,
    )


# --------------------------------------------------------------------------- #
# 单元测试
# --------------------------------------------------------------------------- #
def test_dataset_and_collator(tmp_path):
    files = _make_corpus(tmp_path)
    tok = _make_tokenizer(files)
    ds = H5ADCorpusDataset(files=files, tokenizer=tok)
    assert len(ds) == 2 * 6
    sample = ds[0]
    assert "gene_ids" in sample and sample["gene_ids"].dtype == np.int64

    collate = make_collator(tok)
    batch = collate([ds[i] for i in range(4)])
    assert batch["gene_ids"].shape[0] == 4
    assert batch["gene_ids"].shape[1] <= tok.cfg.max_genes
    assert batch["attention_mask"].dtype == torch.bool
    assert "value_ids" in batch
    assert batch["pos_emb"].shape[-1] == 16


def test_apply_mgm_mask_8010_10():
    gene_ids = torch.arange(8, 8 + 5 * 20).reshape(5, 20)  # 全部为非特殊 id
    attn = torch.ones_like(gene_ids, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    masked, mp = apply_mgm_mask(
        gene_ids,
        attn,
        vocab_size=200,
        mask_ratio=0.5,
        mask_token_prob=0.8,
        random_token_prob=0.1,
        generator=g,
    )
    assert mp.sum() > 0
    # 所有未被 mask 位置必须保持原值
    assert torch.equal(masked[~mp], gene_ids[~mp])


def test_mgm_and_info_nce_shapes():
    B, L, V = 3, 6, 32
    logits = torch.randn(B, L, V, requires_grad=True)
    target = torch.randint(0, V, (B, L))
    mp = torch.zeros(B, L, dtype=torch.bool)
    mp[0, 1] = mp[1, 2] = mp[2, 3] = True
    loss = mgm_loss(logits, target, mp)
    loss.backward()
    assert torch.isfinite(loss)

    z1 = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    nce = info_nce(z1, z2, temperature=0.1)
    assert torch.isfinite(nce) and nce > 0


def test_pretrain_module_one_step(tmp_path):
    files = _make_corpus(tmp_path, n_files=1, n_obs=8)
    tok = _make_tokenizer(files)
    ds = H5ADCorpusDataset(files=files, tokenizer=tok)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=make_collator(tok))

    pcfg = PretrainConfig(
        model_config=_tiny_model_cfg(len(tok.vocab)),
        masking={"mask_ratio": 0.3, "mask_token_prob": 0.8, "random_token_prob": 0.1},
        losses={"mgm_weight": 1.0, "ccl_weight": 0.1, "ccl_temperature": 0.07},
        optim={
            "lr": 1e-3,
            "weight_decay": 0.0,
            "betas": (0.9, 0.95),
            "warmup_steps": 1,
            "max_steps": 3,
        },
    )
    module = SpaFMPretrainModule(pcfg)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        default_root_dir=str(tmp_path / "runs"),
    )
    trainer.fit(module, loader)
    # 跑完不报错且 step 推进
    assert trainer.global_step >= 3


def test_pretrain_loss_decreases(tmp_path):
    files = _make_corpus(tmp_path, n_files=1, n_obs=16, n_vars=16)
    tok = _make_tokenizer(files)
    ds = H5ADCorpusDataset(files=files, tokenizer=tok)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=make_collator(tok))

    pcfg = PretrainConfig(
        model_config=_tiny_model_cfg(len(tok.vocab)),
        masking={"mask_ratio": 0.3, "mask_token_prob": 0.8, "random_token_prob": 0.1},
        losses={"mgm_weight": 1.0, "ccl_weight": 0.0, "ccl_temperature": 0.07},
        optim={
            "lr": 5e-3,
            "weight_decay": 0.0,
            "betas": (0.9, 0.95),
            "warmup_steps": 0,
            "max_steps": 30,
        },
    )
    module = SpaFMPretrainModule(pcfg)
    module.train()

    # 手动跑 30 步看 loss 趋势
    optim_dict = module.configure_optimizers()
    optimizer = optim_dict["optimizer"]
    losses = []
    it = iter(loader)
    for _ in range(30):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        loss = module._compute_losses(batch)["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    # 最后 5 步均值应明显小于前 5 步均值（粗略下降）
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_checkpoint_save_and_load(tmp_path):
    files = _make_corpus(tmp_path, n_files=1, n_obs=4)
    tok = _make_tokenizer(files)
    ds = H5ADCorpusDataset(files=files, tokenizer=tok)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=make_collator(tok))

    pcfg = PretrainConfig(
        model_config=_tiny_model_cfg(len(tok.vocab)),
        optim={"lr": 1e-3, "weight_decay": 0.0, "warmup_steps": 0, "max_steps": 2},
    )
    module = SpaFMPretrainModule(pcfg)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path / "runs"),
    )
    trainer.fit(module, loader)
    ckpt_path = tmp_path / "runs" / "ckpt.pt"
    trainer.save_checkpoint(str(ckpt_path))
    assert ckpt_path.exists()

    # 加载
    loaded = SpaFMPretrainModule.load_from_checkpoint(str(ckpt_path), cfg=pcfg)
    assert loaded.model_cfg.d_model == module.model_cfg.d_model
