"""SpaFM 评测层 —— Stage 7。

提供：

- :mod:`spafm.benchmarks.metrics` — 纯函数指标（acc/f1, ari/nmi, pearson/mse）
- :class:`spafm.benchmarks.embedder.SpaFMEmbedder`
- :class:`spafm.benchmarks.baselines.PCAEmbedder` / :class:`HVGMeanEmbedder`
- :func:`spafm.benchmarks.evaluator.run_benchmark`
"""

from __future__ import annotations

from spafm.benchmarks.baselines import HVGMeanEmbedder, PCAEmbedder
from spafm.benchmarks.ccc import (
    DEFAULT_MOUSE_BRAIN_LR,
    CCCResult,
    extract_outer_attention,
    load_hier_from_ckpt,
    run_ccc_analysis,
)
from spafm.benchmarks.embedder import SpaFMEmbedder
from spafm.benchmarks.evaluator import BenchmarkConfig, run_benchmark
from spafm.benchmarks.metrics import cluster_scores, linear_probe_cv, regression_scores
from spafm.benchmarks.svg import (
    SVGResult,
    extract_inner_attention_picture,
    knn_spatial_weights,
    morans_I_batch,
    run_svg_analysis,
)

__all__ = [
    "BenchmarkConfig",
    "CCCResult",
    "DEFAULT_MOUSE_BRAIN_LR",
    "HVGMeanEmbedder",
    "PCAEmbedder",
    "SVGResult",
    "SpaFMEmbedder",
    "cluster_scores",
    "extract_inner_attention_picture",
    "extract_outer_attention",
    "knn_spatial_weights",
    "linear_probe_cv",
    "load_hier_from_ckpt",
    "morans_I_batch",
    "regression_scores",
    "run_benchmark",
    "run_ccc_analysis",
    "run_svg_analysis",
]
