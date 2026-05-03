"""SpaFM 下游适配模块（Stage 6）。

提供：

- :class:`LoRALinear` 与 :func:`apply_lora` / :func:`mark_only_lora_as_trainable`
- 三种下游 head：cell type / spatial domain / imputation
- :class:`LabeledH5ADDataset` 与 :class:`SpaFMFinetuneModule`
"""

from __future__ import annotations

from spafm.adaptation.dataset import LabeledH5ADDataset
from spafm.adaptation.heads import CellTypeHead, ImputationHead, SpatialDomainHead
from spafm.adaptation.lit_module import FinetuneConfig, SpaFMFinetuneModule
from spafm.adaptation.lora import (
    LoRALinear,
    apply_lora,
    count_trainable,
    mark_only_lora_as_trainable,
)

__all__ = [
    "CellTypeHead",
    "FinetuneConfig",
    "ImputationHead",
    "LabeledH5ADDataset",
    "LoRALinear",
    "SpaFMFinetuneModule",
    "SpatialDomainHead",
    "apply_lora",
    "count_trainable",
    "mark_only_lora_as_trainable",
]
