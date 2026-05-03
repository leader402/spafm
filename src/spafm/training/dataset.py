"""H5ADCorpusDataset：扫描多个 h5ad 文件，按 cell 拼成全局索引。

设计原则：
- AnnData 整文件一次性 read（demo 量很小，几百 MB 内）
- ``__getitem__(i)`` 返回单 cell encode_one 结果（numpy）
- 支持 ``cell_indices`` 子集，便于做 train/val 切分

注意：dataloader num_workers > 0 时，每个 worker 会独立持有 AnnData 引用；
对于较大语料请改写为 backed='r' 模式（v1 再做）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from anndata import read_h5ad
from torch.utils.data import Dataset

from spafm.tokenization import STTokenizer


class H5ADCorpusDataset(Dataset):
    """把多个 ``.h5ad`` 文件中的 cell 拍平成单一索引的数据集。

    Args:
        files: ``.h5ad`` 文件路径列表（或 glob 解析后的结果）。
        tokenizer: 已构建好的 :class:`STTokenizer`。
    """

    def __init__(self, files: list[str | Path], tokenizer: STTokenizer) -> None:
        if not files:
            raise ValueError("H5ADCorpusDataset 至少需要 1 个 h5ad 文件")
        self.files = [Path(f) for f in files]
        self.tokenizer = tokenizer

        # 加载所有 AnnData（demo 体量小）
        self._adatas = [read_h5ad(f) for f in self.files]
        # 预计算 var → token id（每个文件不同）
        self._var_token_ids = [tokenizer._gene_id_array(a) for a in self._adatas]
        # 预计算 spatial 坐标
        self._coords = []
        for a in self._adatas:
            if "spatial" not in a.obsm:
                raise KeyError(f"{a} 缺少 obsm['spatial']")
            self._coords.append(np.asarray(a.obsm["spatial"], dtype=np.float32)[:, :2])

        # 全局索引：(file_idx, row_idx)
        self._index: list[tuple[int, int]] = []
        for fi, a in enumerate(self._adatas):
            self._index.extend((fi, ri) for ri in range(a.n_obs))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        fi, ri = self._index[i]
        adata = self._adatas[fi]
        X = adata.X
        row = X[ri]
        if sp.issparse(row):
            row_dense = np.asarray(row.todense()).ravel()
        else:
            row_dense = np.asarray(row).ravel()
        out = self.tokenizer.encode_one(
            row_counts=row_dense,
            coord=self._coords[fi][ri],
            var_token_ids=self._var_token_ids[fi],
        )
        # encode_one 不返回 pos_emb；推迟到 collate 中统一编码（节省 worker CPU）
        return out
