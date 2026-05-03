"""SpaFM 预训练 CLI。

使用：

    python -m scripts.train_pretrain --config configs/training/spafm-s-pretrain.yaml
    python -m scripts.train_pretrain --config ... --override trainer.max_steps=5 batch_size=2

支持简单的 ``a.b.c=value`` 覆盖语法（值会被 yaml.safe_load 解析）。
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import typer
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from rich.console import Console
from torch.utils.data import DataLoader

from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig
from spafm.training import H5ADCorpusDataset, SpaFMPretrainModule
from spafm.training.collator import make_collator
from spafm.training.lit_module import PretrainConfig

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


# --------------------------------------------------------------------------- #
def _set_nested(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override 必须形如 key=value：{ov}")
        k, v = ov.split("=", 1)
        v_parsed = yaml.safe_load(v)
        _set_nested(cfg, k.strip(), v_parsed)
    return cfg


def _check_tokenization_health(
    dataset,
    tokenizer: STTokenizer,
    *,
    min_maskable_ratio: float = 0.5,
    n_samples: int = 8,
    strict: bool = True,
) -> float:
    """对前 ``n_samples`` 个 cell 计算 *可 mask token 占比*。

    可 mask = 非 special token（PAD/CLS/MASK/UNK/SEP/BOS/EOS/NICHE 之外）。
    比例过低通常意味着 ``vocab_path`` 与数据物种不匹配 → 几乎全 UNK，
    MGM 损失会沦为常数 0（参见 commit c27ae97）。

    返回实测比例；当 ``strict`` 且低于阈值时抛 SystemExit。
    """
    import numpy as _np

    from spafm.training.masking import _SPECIAL_IDS

    n = min(n_samples, len(dataset))
    if n == 0:
        return 0.0
    total = 0
    maskable = 0
    for i in range(n):
        gids = dataset[i]["gene_ids"]
        arr = _np.asarray(gids).ravel()
        total += int(arr.size)
        for sid in _SPECIAL_IDS:
            arr = arr[arr != sid]
        maskable += int(arr.size)
    ratio = maskable / max(total, 1)
    msg = (
        f"[health-check] 词表匹配度：{ratio:.1%} 可 mask "
        f"（{maskable}/{total} token，前 {n} 个 cell；vocab_size={len(tokenizer.vocab)}）"
    )
    if ratio < min_maskable_ratio:
        hint = (
            f"\n[red]✗ 可 mask token 占比 {ratio:.1%} < {min_maskable_ratio:.1%}：\n"
            "  几乎所有 gene 都被 tokenize 成 UNK，MGM 损失会失效。\n"
            "  常见原因：tokenizer vocab_path 与数据物种不匹配。\n"
            "  请检查 configs/tokenizer/*.yaml 的 vocab_path 是否指向\n"
            "  ingest 阶段为该 corpus 扩展的 gene_vocab_*.tsv。\n"
            "  如确认无误，可在训练 yaml 中显式设 strict_tokenization_check: false。"
        )
        console.log(f"[red]{msg}")
        if strict:
            raise SystemExit(hint)
    else:
        console.log(f"[green]{msg}")
    return ratio


def _build_tokenizer(tok_cfg_path: str, h5ads: list[Path]) -> STTokenizer:
    tcfg = TokenizerConfig.from_yaml(tok_cfg_path)
    if tcfg.vocab_path and Path(tcfg.vocab_path).exists():
        vocab = GeneVocab.from_tsv(tcfg.vocab_path)
    else:
        # demo：从 corpus 即时建词表
        from anndata import read_h5ad

        symbols: list[str] = []
        seen = set()
        for f in h5ads:
            a = read_h5ad(f)
            col = "gene_symbol" if "gene_symbol" in a.var.columns else None
            names = a.var[col].astype(str).tolist() if col else a.var_names.astype(str).tolist()
            for s in names:
                if s not in seen:
                    seen.add(s)
                    symbols.append(s)
        vocab = GeneVocab.from_symbols(symbols)
        console.log(f"[yellow]即时构建词表：{len(vocab)} 个 token（含 special）")
    return STTokenizer(vocab=vocab, cfg=tcfg)


# --------------------------------------------------------------------------- #
@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="顶层训练 yaml"),
    override: list[str] = typer.Option(
        None, "--override", "-o", help="key=value 形式的覆盖（可多个）"
    ),
) -> None:
    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    raw = _apply_overrides(raw, override or [])
    console.print(raw)

    pl.seed_everything(int(raw.get("seed", 42)), workers=True)

    # 数据
    glob_pat = raw["data"]["h5ad_glob"]
    files = sorted(Path(p) for p in _glob.glob(glob_pat))
    if not files:
        raise SystemExit(f"❌ 没有匹配到 h5ad 文件：{glob_pat}")
    console.log(f"匹配到 {len(files)} 个 h5ad 文件")

    tokenizer = _build_tokenizer(raw["data"]["tokenizer_config"], files)

    # 把词表大小写回 model_config（如果 yaml 里没显式指定的话保持 64000，
    # 实际 forward 时只要 ≥ 出现的最大 id 就行；这里用 max(64000, len(vocab)) 兜底）
    mc_path = raw["model_config"]
    from spafm.models import ModelConfig

    model_cfg = (
        ModelConfig.from_yaml(mc_path)
        if isinstance(mc_path, (str, Path))
        else ModelConfig(**mc_path)
    )
    if len(tokenizer.vocab) > model_cfg.vocab_size:
        console.log(f"[yellow]vocab_size 自动放大 {model_cfg.vocab_size} → {len(tokenizer.vocab)}")
        model_cfg.vocab_size = len(tokenizer.vocab)

    dataset = H5ADCorpusDataset(files=list(files), tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=int(raw.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(raw.get("num_workers", 0)),
        collate_fn=make_collator(tokenizer),
        drop_last=False,
    )

    # 健康检查：词表 vs 数据匹配度（防 MGM mask 全 0 的隐性 bug）
    _check_tokenization_health(
        dataset,
        tokenizer,
        min_maskable_ratio=float(raw.get("min_maskable_ratio", 0.5)),
        strict=bool(raw.get("strict_tokenization_check", True)),
    )

    # 模块
    pcfg = PretrainConfig.from_dict(raw)
    pcfg.model_config = model_cfg
    module = SpaFMPretrainModule(pcfg)

    # 可选：知识对齐（Stage 5）
    k_cfg = raw.get("knowledge", {}) or {}
    if k_cfg.get("enabled", False):
        from spafm.knowledge import GenePriorBank, PriorAligner

        prior_path = k_cfg["prior_path"]
        fmt = k_cfg.get("prior_format", "npz")
        bank = (
            GenePriorBank.from_npz(prior_path)
            if fmt == "npz"
            else GenePriorBank.from_tsv(prior_path)
        )
        prior_mat, prior_mask = bank.align_to_vocab(tokenizer.vocab)
        # 对齐到模型 vocab_size：当模型词表更大时，把 prior 行 0 填充并 mask=False
        V_model = model_cfg.vocab_size
        V_tok = prior_mat.shape[0]
        if V_model > V_tok:
            import torch as _t

            pad_mat = _t.zeros((V_model - V_tok, prior_mat.shape[1]), dtype=prior_mat.dtype)
            pad_mask = _t.zeros((V_model - V_tok,), dtype=prior_mask.dtype)
            prior_mat = _t.cat([prior_mat, pad_mat], dim=0)
            prior_mask = _t.cat([prior_mask, pad_mask], dim=0)
        elif V_model < V_tok:
            prior_mat = prior_mat[:V_model]
            prior_mask = prior_mask[:V_model]
        console.log(f"[cyan]知识先验：{bank.dim} 维，命中 {int(prior_mask.sum())}/{V_model}")
        aligner = PriorAligner(
            d_model=model_cfg.d_model,
            d_prior=bank.dim,
            prior_matrix=prior_mat,
            prior_mask=prior_mask,
            freeze_prior=bool(k_cfg.get("freeze_prior", True)),
        )
        module.attach_prior_aligner(aligner)

    # Trainer
    tcfg = raw.get("trainer", {})
    ckpt = ModelCheckpoint(
        dirpath=str(Path(tcfg.get("default_root_dir", "runs/spafm-s")) / "ckpt"),
        save_top_k=1,
        every_n_train_steps=max(1, int(tcfg.get("max_steps", 1000)) // 2),
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator=tcfg.get("accelerator", "auto"),
        devices=tcfg.get("devices", 1),
        precision=tcfg.get("precision", 32),
        max_steps=int(tcfg.get("max_steps", 1000)),
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 10)),
        gradient_clip_val=float(tcfg.get("gradient_clip_val", 0.0)) or None,
        default_root_dir=tcfg.get("default_root_dir", "runs/spafm-s"),
        callbacks=[ckpt],
        enable_progress_bar=True,
    )
    trainer.fit(module, loader)
    console.print("[bold green]✓ 训练结束")


if __name__ == "__main__":
    app()
