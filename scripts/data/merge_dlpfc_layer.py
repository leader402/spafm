"""把 spatialLIBD DLPFC 的层级真值合并到 processed h5ad。

来源：``data/raw/spatiallibd-dlpfc-maynard2021/barcode_level_layer_map.tsv``

格式（无 header，制表符分隔）：

    barcode<TAB>sample_id<TAB>layer

其中 ``layer ∈ {L1, L2, L3, L4, L5, L6, WM}``。

行为：
- 遍历 ``data/processed/spatiallibd-dlpfc-maynard2021/<sample_id>.h5ad``；
- 用 barcode 取 layer，写入 ``obs['layer']``（string）；
- 若 ``obs['niche_label']`` 已存在但全为空串，则把 ``layer`` 覆盖进去
  （兼容现有 benchmark 配置 label_key=niche_label）；
- 标记 ``uns['spafm_layer_source'] = 'spatialLIBD-Maynard2021'``；
- 默认 in-place 覆盖；可用 ``--dry-run`` 只打印。

用法
----
::

    python scripts/data/merge_dlpfc_layer.py \\
        --tsv data/raw/spatiallibd-dlpfc-maynard2021/barcode_level_layer_map.tsv \\
        --processed-dir data/processed/spatiallibd-dlpfc-maynard2021
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from anndata import read_h5ad
from rich.console import Console

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


@app.command()
def main(
    tsv: Path = typer.Option(
        Path("data/raw/spatiallibd-dlpfc-maynard2021/barcode_level_layer_map.tsv"),
        "--tsv",
        help="层级真值 TSV（barcode\\tsample_id\\tlayer）",
    ),
    processed_dir: Path = typer.Option(
        Path("data/processed/spatiallibd-dlpfc-maynard2021"),
        "--processed-dir",
        help="processed h5ad 目录",
    ),
    overwrite_niche: bool = typer.Option(
        True,
        "--overwrite-niche/--no-overwrite-niche",
        help="若 obs['niche_label'] 全空，则用 layer 覆盖（默认开）",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="不写盘，只打印"),
) -> None:
    if not tsv.exists():
        raise SystemExit(f"❌ 找不到 TSV: {tsv}")
    if not processed_dir.exists():
        raise SystemExit(f"❌ 找不到 processed 目录: {processed_dir}")

    df = pd.read_csv(tsv, sep="\t", header=None, names=["barcode", "sample_id", "layer"])
    df["sample_id"] = df["sample_id"].astype(str)
    console.log(f"TSV 行数 = {len(df)}, 层级类别 = {sorted(df['layer'].unique())}")

    # 按 sample_id 索引
    by_sample = {sid: g.set_index("barcode")["layer"] for sid, g in df.groupby("sample_id")}

    files = sorted(processed_dir.glob("*.h5ad"))
    if not files:
        raise SystemExit(f"❌ 没有 h5ad: {processed_dir}/*.h5ad")

    summary: list[dict] = []
    for f in files:
        sid = f.stem  # e.g. 151507
        if sid not in by_sample:
            console.log(f"[yellow]跳过 {f.name}（TSV 中无对应 sample_id={sid}）")
            continue

        a = read_h5ad(f)
        bc = a.obs_names.astype(str)
        lookup = by_sample[sid]
        # 直接 reindex
        layer = lookup.reindex(bc).fillna("").astype(str)
        n_hit = int((layer != "").sum())
        n_total = len(bc)

        a.obs["layer"] = layer.values
        if overwrite_niche:
            cur = a.obs.get("niche_label")
            cur_empty = cur is None or (cur.astype(str).str.len() == 0).all()
            if cur_empty:
                a.obs["niche_label"] = layer.values
        a.uns["spafm_layer_source"] = "spatialLIBD-Maynard2021"

        layer_counts = layer[layer != ""].value_counts().to_dict()
        summary.append(
            {
                "file": f.name,
                "sample_id": sid,
                "n_obs": n_total,
                "n_with_layer": n_hit,
                "coverage": round(n_hit / max(1, n_total), 4),
                "layers": dict(sorted(layer_counts.items())),
            }
        )

        if not dry_run:
            a.write_h5ad(f)
            console.log(
                f"[green]✓ {f.name}: {n_hit}/{n_total} = {n_hit / n_total:.1%} 已写入 obs['layer']"
            )
        else:
            console.log(
                f"[cyan][dry-run] {f.name}: {n_hit}/{n_total} = {n_hit / n_total:.1%}"
            )

    console.rule("Summary")
    for row in summary:
        console.print(row)
    total_obs = sum(r["n_obs"] for r in summary)
    total_hit = sum(r["n_with_layer"] for r in summary)
    console.print(
        f"[bold]Total:[/] {total_hit}/{total_obs} = "
        f"{total_hit / max(1, total_obs):.1%} spots 拿到 layer 标签"
    )


if __name__ == "__main__":
    app()
