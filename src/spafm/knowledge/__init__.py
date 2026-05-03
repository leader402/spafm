"""SpaFM 知识增强模块（Stage 5）。

把外部基因先验（如 GenePT/GO 嵌入）对齐到模型的 :class:`GeneEmbedding`。
"""

from __future__ import annotations

from spafm.knowledge.aligner import PriorAligner, alignment_loss
from spafm.knowledge.gene_priors import GenePriorBank
from spafm.knowledge.synth import build_synthetic_prior

__all__ = [
    "GenePriorBank",
    "PriorAligner",
    "alignment_loss",
    "build_synthetic_prior",
]
