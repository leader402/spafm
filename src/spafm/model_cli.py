"""``spafm model`` 子命令。"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from spafm.registry import MODEL_REGISTRY, get_model_card

app = typer.Typer(no_args_is_help=True, help="模型注册表 / 下载相关")
console = Console()


@app.command("list")
def list_models_cmd() -> None:
    """列出已登记的模型。"""
    table = Table(title=f"已登记模型（共 {len(MODEL_REGISTRY)}）")
    for col in ("ID", "Size", "Params", "Pretrain Data", "License", "Status"):
        table.add_column(col)
    for c in MODEL_REGISTRY.values():
        table.add_row(c.id, c.size, f"{c.n_params:,}", c.pretraining_data, c.license, c.status)
    console.print(table)


@app.command("info")
def info_cmd(model_id: str = typer.Argument(..., help="模型 ID，例如 spafm-s-v0")) -> None:
    """打印一个模型的完整 ModelCard。"""
    c = get_model_card(model_id)
    for k, v in c.__dict__.items():
        console.print(f"[cyan]{k:18s}[/cyan] {v}")
    console.print(f"[cyan]{'status':18s}[/cyan] {c.status}")
