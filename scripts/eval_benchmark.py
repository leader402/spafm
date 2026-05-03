"""SpaFM 评测 CLI。

用法:

    python -m scripts.eval_benchmark -c configs/benchmark/spafm-s-eval.yaml
    python -m scripts.eval_benchmark -c ... -o embedder.type=pca
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from spafm.benchmarks import BenchmarkConfig, run_benchmark

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


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c"),
    override: list[str] = typer.Option(None, "--override", "-o"),
) -> None:
    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    raw = _apply_overrides(raw, override or [])
    cfg = BenchmarkConfig(
        data=raw.get("data", {}),
        embedder=raw.get("embedder", {}),
        tasks=raw.get("tasks", []),
        output=raw.get("output", {}),
        seed=int(raw.get("seed", 42)),
    )
    out = run_benchmark(cfg)

    table = Table(title=f"SpaFM Benchmark · embedder={out['config']['embedder']}")
    for col in ("embedder", "task", "metric", "value"):
        table.add_column(col)
    for r in out["results"]:
        v = r["value"]
        v_str = f"{v:.4f}" if isinstance(v, float) and v == v else str(v)  # NaN safe
        table.add_row(r["embedder"], r["task"], r["metric"], v_str)
    console.print(table)
    if cfg.output.get("json_path"):
        console.log(f"[green]结果已写入 {cfg.output['json_path']}")


if __name__ == "__main__":
    app()
