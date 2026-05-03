"""SpaFM scaling 报告 CLI。

用法:
    python -m scripts.scaling_report
    python -m scripts.scaling_report --fit results.json
    python -m scripts.scaling_report --tokens 1e10
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from spafm.scaling import (
    SIZE_CONFIGS,
    estimate_flops_per_token,
    estimate_params_from_cfg,
    estimate_total_flops,
    fit_scaling_law,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


def _human(n: float) -> str:
    for unit, base in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= base:
            return f"{n / base:.2f}{unit}"
    return f"{n:.0f}"


@app.command()
def main(
    fit: Path = typer.Option(None, "--fit", help="JSON 文件 [{params,loss}]，做 scaling-law 拟合"),
    tokens: float = typer.Option(0.0, "--tokens", help="若 >0，给出每档训练总 FLOPs 估计"),
) -> None:
    """打印 S/M/L 三档参数 / FLOPs，并可选拟合 scaling law。"""
    table = Table(title="SpaFM scaling 报告")
    cols = ["size", "d_model", "n_layers", "vocab", "params(est)", "FLOPs/token"]
    if tokens > 0:
        cols.append(f"total FLOPs @{_human(tokens)} tok")
    for c in cols:
        table.add_column(c)
    for name, cfg in SIZE_CONFIGS.items():
        parts = estimate_params_from_cfg(cfg)
        fpt = estimate_flops_per_token(cfg)
        row = [
            name,
            str(cfg.d_model),
            str(cfg.n_layers),
            str(cfg.vocab_size),
            _human(parts["total"]),
            _human(fpt),
        ]
        if tokens > 0:
            row.append(_human(estimate_total_flops(cfg, tokens)))
        table.add_row(*row)
    console.print(table)

    if fit:
        raw = json.loads(Path(fit).read_text(encoding="utf-8"))
        points = [(float(r["params"]), float(r["loss"])) for r in raw]
        result = fit_scaling_law(points)
        console.rule("Scaling-law 拟合")
        console.print(
            f"[cyan]L(P) = A · P^(-alpha)[/cyan]    "
            f"alpha={result.alpha:.4f}  A={result.A:.4g}  R²={result.r2:.4f}"
        )


if __name__ == "__main__":
    app()
