"""SpaFM 下游微调 CLI。

用法:

    python -m scripts.train_finetune -c configs/training/spafm-s-finetune.yaml
    python -m scripts.train_finetune -c ... -o adaptation.strategy=linear_probe
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

from spafm.adaptation import FinetuneConfig, LabeledH5ADDataset, SpaFMFinetuneModule
from spafm.adaptation.dataset import labeled_collate
from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig
from spafm.training import H5ADCorpusDataset
from spafm.training.collator import make_collator

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


def _set_nested(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override 必须形如 key=value：{ov}")
        k, v = ov.split("=", 1)
        _set_nested(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def _build_tokenizer(tok_cfg_path: str, h5ads: list[Path]) -> STTokenizer:
    tcfg = TokenizerConfig.from_yaml(tok_cfg_path)
    if tcfg.vocab_path and Path(tcfg.vocab_path).exists():
        vocab = GeneVocab.from_tsv(tcfg.vocab_path)
    else:
        from anndata import read_h5ad

        symbols: list[str] = []
        seen: set[str] = set()
        for f in h5ads:
            a = read_h5ad(f)
            col = "gene_symbol" if "gene_symbol" in a.var.columns else None
            names = a.var[col].astype(str).tolist() if col else a.var_names.astype(str).tolist()
            for s in names:
                if s not in seen:
                    seen.add(s)
                    symbols.append(s)
        vocab = GeneVocab.from_symbols(symbols)
        console.log(f"[yellow]即时构建词表：{len(vocab)} tokens")
    return STTokenizer(vocab=vocab, cfg=tcfg)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c"),
    override: list[str] = typer.Option(None, "--override", "-o"),
) -> None:
    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    raw = _apply_overrides(raw, override or [])
    console.print(raw)

    pl.seed_everything(int(raw.get("seed", 42)), workers=True)

    files = sorted(Path(p) for p in _glob.glob(raw["data"]["h5ad_glob"]))
    if not files:
        raise SystemExit(f"❌ 未匹配 h5ad：{raw['data']['h5ad_glob']}")
    console.log(f"匹配到 {len(files)} 个 h5ad")

    tokenizer = _build_tokenizer(raw["data"]["tokenizer_config"], files)

    # 自动放大 model vocab_size
    from spafm.models import ModelConfig

    mc_path = raw["model_config"]
    model_cfg = (
        ModelConfig.from_yaml(mc_path)
        if isinstance(mc_path, (str, Path))
        else ModelConfig(**mc_path)
    )
    if len(tokenizer.vocab) > model_cfg.vocab_size:
        console.log(f"[yellow]vocab_size {model_cfg.vocab_size} → {len(tokenizer.vocab)}")
        model_cfg.vocab_size = len(tokenizer.vocab)

    task = raw["data"].get("task", "classification")
    base_collate = make_collator(tokenizer)
    if task == "classification":
        ds = LabeledH5ADDataset(
            files=list(files),
            tokenizer=tokenizer,
            label_key=raw["data"]["label_key"],
        )
        # 真正的类别数 ← 数据
        raw["head"]["num_classes"] = ds.num_classes
        loader = DataLoader(
            ds,
            batch_size=int(raw.get("batch_size", 8)),
            shuffle=True,
            num_workers=int(raw.get("num_workers", 0)),
            collate_fn=lambda b: labeled_collate(b, base_collate),
            drop_last=False,
        )
    elif task == "imputation":
        ds = H5ADCorpusDataset(files=list(files), tokenizer=tokenizer)
        loader = DataLoader(
            ds,
            batch_size=int(raw.get("batch_size", 8)),
            shuffle=True,
            num_workers=int(raw.get("num_workers", 0)),
            collate_fn=base_collate,
        )
    else:
        raise ValueError(f"未知 task: {task}")

    fcfg = FinetuneConfig.from_dict(raw)
    fcfg.model_config = model_cfg
    module = SpaFMFinetuneModule(fcfg)

    tcfg = raw.get("trainer", {})
    callbacks = [
        ModelCheckpoint(
            dirpath=str(Path(tcfg.get("default_root_dir", "runs/spafm-s-ft")) / "ckpt"),
            save_top_k=1,
            every_n_train_steps=max(1, int(tcfg.get("max_steps", 200)) // 2),
            save_last=True,
        )
    ]
    trainer = pl.Trainer(
        accelerator=tcfg.get("accelerator", "auto"),
        devices=tcfg.get("devices", 1),
        precision=tcfg.get("precision", 32),
        max_steps=int(tcfg.get("max_steps", 200)),
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 10)),
        gradient_clip_val=float(tcfg.get("gradient_clip_val", 0.0)) or None,
        default_root_dir=tcfg.get("default_root_dir", "runs/spafm-s-ft"),
        callbacks=callbacks,
    )
    trainer.fit(module, loader)
    console.print("[bold green]✓ 微调结束")


if __name__ == "__main__":
    app()
