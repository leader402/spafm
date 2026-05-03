"""SpaFM Hierarchical 预训练 CLI。

使用：

    spafm pretrain-hier --config configs/training/spafm-s-pretrain-hier-mouse.yaml
    spafm pretrain-hier -c <yaml> -o trainer.max_steps=2 -o num_workers=0
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path

import lightning.pytorch as pl
import typer
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from rich.console import Console
from torch.utils.data import DataLoader

from scripts.train_pretrain import (
    _apply_overrides,
    _build_tokenizer,
    _check_tokenization_health,
)
from spafm.models.hierarchical import HierarchicalConfig
from spafm.models.spafm import ModelConfig
from spafm.training import (
    H5ADCorpusDataset,
    HierarchicalSpaFMPretrainModule,
    HierPretrainConfig,
    SliceDataset,
    make_slice_collator,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


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

    # data.h5ad_glob 兼容三种写法：
    #   - 单字符串                              （单数据集 / 早期 config）
    #   - list[str]                             （多数据集，按顺序合并 + 去重）
    #   - data.h5ad_globs: list[str]            （别名，更直观）
    raw_globs: list[str] = []
    glob_field = raw["data"].get("h5ad_globs") or raw["data"].get("h5ad_glob")
    if glob_field is None:
        raise SystemExit("❌ data 缺少 h5ad_glob 或 h5ad_globs")
    if isinstance(glob_field, str):
        raw_globs = [glob_field]
    elif isinstance(glob_field, list):
        raw_globs = [str(g) for g in glob_field]
    else:
        raise SystemExit(f"❌ data.h5ad_glob 非法类型：{type(glob_field)}")

    seen: set[Path] = set()
    files: list[Path] = []
    for pat in raw_globs:
        matched = sorted(Path(p) for p in _glob.glob(pat))
        for p in matched:
            if p not in seen:
                seen.add(p)
                files.append(p)
    if not files:
        raise SystemExit(f"❌ 没有匹配到 h5ad 文件：{raw_globs}")
    console.log(f"匹配到 {len(files)} 个 h5ad 文件（来自 {len(raw_globs)} 个 glob）")

    tokenizer = _build_tokenizer(raw["data"]["tokenizer_config"], files)

    # 健康检查复用 H5ADCorpusDataset
    health_ds = H5ADCorpusDataset(files=list(files), tokenizer=tokenizer)
    _check_tokenization_health(
        health_ds,
        tokenizer,
        min_maskable_ratio=float(raw.get("min_maskable_ratio", 0.5)),
        strict=bool(raw.get("strict_tokenization_check", True)),
    )

    # 解析 model config，并按词表大小放大 inner.vocab_size
    mc_path = raw["model_config"]
    if isinstance(mc_path, (str, Path)):
        hier_cfg = HierarchicalConfig.from_yaml(mc_path)
    elif isinstance(mc_path, dict):
        inner = mc_path.get("inner", {})
        rest = {k: v for k, v in mc_path.items() if k != "inner"}
        hier_cfg = HierarchicalConfig(inner=ModelConfig(**inner), **rest)
    else:
        hier_cfg = HierarchicalConfig()

    if len(tokenizer.vocab) > hier_cfg.inner.vocab_size:
        console.log(
            f"[yellow]inner.vocab_size 自动放大 {hier_cfg.inner.vocab_size} → {len(tokenizer.vocab)}"
        )
        hier_cfg.inner.vocab_size = len(tokenizer.vocab)

    # SliceDataset + slice collator
    dcfg = raw["data"]
    n_spots = int(dcfg.get("n_spots_per_sample", 64))
    samples_per_slice = int(dcfg.get("samples_per_slice", 16))
    dataset = SliceDataset(
        files=list(files),
        tokenizer=tokenizer,
        n_spots_per_sample=n_spots,
        samples_per_slice=samples_per_slice,
        seed=int(raw.get("seed", 42)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(raw.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(raw.get("num_workers", 0)),
        collate_fn=make_slice_collator(tokenizer, n_spots),
        drop_last=False,
    )

    pcfg = HierPretrainConfig.from_dict(raw)
    pcfg.model_config = hier_cfg
    module = HierarchicalSpaFMPretrainModule(pcfg)

    tcfg = raw.get("trainer", {})
    ckpt = ModelCheckpoint(
        dirpath=str(Path(tcfg.get("default_root_dir", "runs/spafm-s-hier")) / "ckpt"),
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
        default_root_dir=tcfg.get("default_root_dir", "runs/spafm-s-hier"),
        callbacks=[ckpt],
        enable_progress_bar=True,
    )
    trainer.fit(module, loader)
    console.print("[bold green]✓ Hierarchical 预训练结束")


if __name__ == "__main__":
    app()
