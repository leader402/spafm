"""CCC（细胞-细胞通讯）评测 CLI。

用法：

    spafm eval-ccc \\
        -h data/processed/visium-mouse-brain-sagittal-posterior-10x/V1_Mouse_Brain_Sagittal_Posterior.h5ad \\
        --ckpt runs/spafm-s-hier-mouse/ckpt/last.ckpt \\
        --model-config configs/model/spafm-s-hier.yaml \\
        --tokenizer-config configs/tokenizer/spafm-s-visium-mouse.yaml \\
        --label-key cell_type \\
        --max-spots 400 \\
        --device cpu \\
        --out runs/ccc/post.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from scripts.train_pretrain import _build_tokenizer
from spafm.benchmarks.ccc import load_hier_from_ckpt, run_ccc_analysis

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


@app.command()
def main(
    h5ad: Path = typer.Option(..., "-h", "--h5ad", help="单切片 h5ad"),
    ckpt: Path = typer.Option(..., "--ckpt", help="HierarchicalSpaFM ckpt"),
    model_config: Path = typer.Option(..., "--model-config", help="hierarchical 模型 yaml"),
    tokenizer_config: Path = typer.Option(..., "--tokenizer-config", help="tokenizer yaml"),
    label_key: str = typer.Option("cell_type", "--label-key"),
    max_spots: int = typer.Option(400, "--max-spots"),
    device: str = typer.Option("cpu", "--device"),
    seed: int = typer.Option(0, "--seed"),
    out: Path | None = typer.Option(None, "--out", help="结果 json 输出路径"),
) -> None:
    tokenizer = _build_tokenizer(str(tokenizer_config), [h5ad])
    console.log(f"vocab_size={len(tokenizer.vocab)}")

    model = load_hier_from_ckpt(
        ckpt_path=ckpt,
        model_config=model_config,
        vocab_size_override=None,
        device=device,
    )

    res = run_ccc_analysis(
        h5ad=h5ad,
        model=model,
        tokenizer=tokenizer,
        label_key=label_key,
        max_spots=max_spots,
        device=device,
        seed=seed,
    )

    payload = res.to_dict()
    console.print(payload)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        console.log(f"[green]✓ 已写入 {out}")


if __name__ == "__main__":
    app()
