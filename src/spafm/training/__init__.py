"""SpaFM 训练层 —— Stage 4 预训练。

提供：

- :class:`H5ADCorpusDataset`：扫描 ``*.h5ad`` 并按 cell 拼成全局索引
- :func:`mgm_collate`：动态 padding 的 collate_fn
- :func:`apply_mgm_mask`：BERT 风格的 80/10/10 mask 策略
- :func:`mgm_loss` / :func:`info_nce`
- :class:`SpaFMPretrainModule`：Lightning 训练模块
"""

from __future__ import annotations

from spafm.training.collator import mgm_collate
from spafm.training.dataset import H5ADCorpusDataset
from spafm.training.hier_lit_module import (
    HierarchicalSpaFMPretrainModule,
    HierPretrainConfig,
)
from spafm.training.lit_module import PretrainConfig, SpaFMPretrainModule
from spafm.training.losses import info_nce, mgm_loss
from spafm.training.masking import apply_mgm_mask
from spafm.training.slice_dataset import SliceDataset, make_slice_collator

__all__ = [
    "H5ADCorpusDataset",
    "HierPretrainConfig",
    "HierarchicalSpaFMPretrainModule",
    "PretrainConfig",
    "SliceDataset",
    "SpaFMPretrainModule",
    "apply_mgm_mask",
    "info_nce",
    "make_slice_collator",
    "mgm_collate",
    "mgm_loss",
]
