"""LabeledH5ADDataset：在 H5ADCorpusDataset 基础上同时返回 cell-level label。

label 来源 ``adata.obs[label_key]``，按字符串值整数化（label_to_id 全数据集复用）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spafm.tokenization import STTokenizer
from spafm.training import H5ADCorpusDataset


class LabeledH5ADDataset(H5ADCorpusDataset):
    """有监督版本的 H5ADCorpusDataset。"""

    def __init__(
        self,
        files: list[str | Path],
        tokenizer: STTokenizer,
        label_key: str,
        label_to_id: dict[str, int] | None = None,
    ) -> None:
        super().__init__(files=files, tokenizer=tokenizer)
        self.label_key = label_key

        # 收集所有标签 → 建立 label_to_id（如未提供）
        if label_to_id is None:
            seen: dict[str, int] = {}
            for a in self._adatas:
                if label_key not in a.obs.columns:
                    raise KeyError(f"adata.obs 缺少 {label_key!r} 列")
                for v in a.obs[label_key].astype(str).tolist():
                    if v not in seen:
                        seen[v] = len(seen)
            self.label_to_id = seen
        else:
            self.label_to_id = dict(label_to_id)

        # 预计算每个全局索引的 label id
        self._labels = np.zeros(len(self._index), dtype=np.int64)
        for gi, (fi, ri) in enumerate(self._index):
            v = str(self._adatas[fi].obs[label_key].iloc[ri])
            self._labels[gi] = self.label_to_id[v]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_id)

    def __getitem__(self, i: int) -> dict:
        out = super().__getitem__(i)
        out["label"] = np.int64(self._labels[i])
        return out


def labeled_collate(batch, base_collator):
    """在 base_collator 输出基础上加上 label tensor。"""
    import torch

    labels = np.asarray([item["label"] for item in batch], dtype=np.int64)
    out = base_collator(batch)
    out["label"] = torch.from_numpy(labels)
    return out
