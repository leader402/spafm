"""SpaFM 顶层 CLI 入口（Typer）。

聚合 5 个子命令组：

- ``spafm version``       打印版本
- ``spafm data ...``      数据语料构建
- ``spafm pretrain``      预训练
- ``spafm finetune``      下游微调
- ``spafm eval``          评测基准
- ``spafm model ...``     模型注册表

老的 ``python -m scripts.train_pretrain ...`` 用法仍然可用。
"""

from __future__ import annotations

import typer

from spafm import __version__
from spafm.data.cli import app as data_app
from spafm.model_cli import app as model_app

app = typer.Typer(
    name="spafm",
    help="SpaFM — Spatial Transcriptomics Foundation Model 命令行入口",
    no_args_is_help=True,
)


@app.command("version")
def version_cmd() -> None:
    """打印 SpaFM 版本。"""
    typer.echo(__version__)


app.add_typer(data_app, name="data", help="数据语料构建相关子命令")
app.add_typer(model_app, name="model", help="模型注册表相关子命令")

# 训练 / 微调 / 评测 —— 直接复用 scripts 里的 main 函数挂成单命令
try:
    from scripts.train_pretrain import main as _pretrain_main

    app.command("pretrain", help="预训练（同 scripts.train_pretrain）")(_pretrain_main)
except Exception:  # noqa: BLE001
    pass
try:
    from scripts.train_pretrain_hier import main as _pretrain_hier_main

    app.command(
        "pretrain-hier",
        help="Hierarchical 预训练（同 scripts.train_pretrain_hier）",
    )(_pretrain_hier_main)
except Exception:  # noqa: BLE001
    pass
try:
    from scripts.train_finetune import main as _finetune_main

    app.command("finetune", help="下游微调（同 scripts.train_finetune）")(_finetune_main)
except Exception:  # noqa: BLE001
    pass
try:
    from scripts.eval_benchmark import main as _eval_main

    app.command("eval", help="评测基准（同 scripts.eval_benchmark）")(_eval_main)
except Exception:  # noqa: BLE001
    pass
try:
    from scripts.eval_ccc import main as _eval_ccc_main

    app.command("eval-ccc", help="CCC 细胞通讯评测（基于 outer attention）")(_eval_ccc_main)
except Exception:  # noqa: BLE001
    pass
try:
    from scripts.eval_svg import main as _eval_svg_main

    app.command("eval-svg", help="SVG 空间可变基因评测（基于内层注意力）")(_eval_svg_main)
except Exception:  # noqa: BLE001
    pass

if __name__ == "__main__":
    app()
