"""SVG（空间可变基因）评测 CLI。

用法：

    spafm eval-svg \\
        -h data/processed/visium-mouse-brain-sagittal-posterior-10x/V1_Mouse_Brain_Sagittal_Posterior.h5ad \\
        --ckpt runs/spafm-s-hier-mouse/ckpt/last.ckpt \\
        --model-config configs/model/spafm-s-hier.yaml \\
        --tokenizer-config configs/tokenizer/spafm-s-visium-mouse.yaml \\
        --max-spots 300 --knn 8 --device cpu \\
        --out runs/svg/posterior.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from scripts.train_pretrain import _build_tokenizer
from spafm.benchmarks.ccc import load_hier_from_ckpt
from spafm.benchmarks.svg import run_svg_analysis

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


@app.command()
def main(
    h5ad: Path = typer.Option(..., "-h", "--h5ad", help="单切片 h5ad"),
    ckpt: Path = typer.Option(..., "--ckpt", help="HierarchicalSpaFM ckpt"),
    model_config: Path = typer.Option(..., "--model-config"),
    tokenizer_config: Path = typer.Option(..., "--tokenizer-config"),
    max_spots: int = typer.Option(300, "--max-spots"),
    knn: int = typer.Option(8, "--knn"),
    top_k: list[int] = typer.Option([20, 50, 100], "--top-k"),
    device: str = typer.Option("cpu", "--device"),
    seed: int = typer.Option(0, "--seed"),
    min_nonnan_frac: float = typer.Option(0.3, "--min-nonnan-frac"),
    chunk_size: int = typer.Option(32, "--chunk-size", help="每次前向多少 spot（控制显存）"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    tokenizer = _build_tokenizer(str(tokenizer_config), [h5ad])
    console.log(f"vocab_size={len(tokenizer.vocab)}")

    model = load_hier_from_ckpt(
        ckpt_path=ckpt,
        model_config=model_config,
        vocab_size_override=None,
        device=device,
    )

    res = run_svg_analysis(
        h5ad=h5ad,
        model=model,
        tokenizer=tokenizer,
        max_spots=max_spots,
        knn=knn,
        top_ks=tuple(top_k),
        device=device,
        seed=seed,
        min_nonnan_frac=min_nonnan_frac,
        chunk_size=chunk_size,
    )

    payload = res.to_dict()
    console.print(payload)
    console.print(f"[cyan]top-10 attention genes:[/] {res.top_genes_attn[:10]}")
    console.print(f"[cyan]top-10 expression genes:[/] {res.top_genes_expr[:10]}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        console.log(f"[green]✓ 已写入 {out}")


if __name__ == "__main__":
    app()
