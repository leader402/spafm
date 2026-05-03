"""GenePriorBank：基因符号 → 先验向量。

支持两种存储格式：

- ``.npz``：``{symbols: (N,) U-string, vectors: (N, d) float32}``
- ``.tsv``：``symbol\\t v0\\t v1\\t ...``（首行为 header；symbol 列名 ``symbol`` 或 ``gene_symbol``）

提供 :meth:`align_to_vocab` 把先验与 :class:`spafm.tokenization.GeneVocab` 对齐，
返回 ``(V, d)`` 张量与 ``(V,)`` bool mask（True=该基因有先验）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from spafm.tokenization.gene_vocab import GeneVocab


@dataclass
class GenePriorBank:
    """``symbol`` → 先验向量。"""

    symbols: list[str]
    vectors: np.ndarray  # (N, d) float32

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.symbols)

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #
    @classmethod
    def from_npz(cls, path: str | Path) -> GenePriorBank:
        data = np.load(path, allow_pickle=False)
        if "symbols" not in data or "vectors" not in data:
            raise KeyError(f"{path} 必须包含 'symbols' 与 'vectors' 两个数组")
        symbols = [str(s).upper() for s in data["symbols"]]
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        if vectors.shape[0] != len(symbols):
            raise ValueError(f"symbols/vectors 行数不一致：{len(symbols)} vs {vectors.shape[0]}")
        return cls(symbols=symbols, vectors=vectors)

    @classmethod
    def from_tsv(cls, path: str | Path) -> GenePriorBank:
        import pandas as pd

        df = pd.read_csv(path, sep="\t")
        sym_col = "symbol" if "symbol" in df.columns else "gene_symbol"
        if sym_col not in df.columns:
            raise KeyError("TSV 必须包含 'symbol' 或 'gene_symbol' 列")
        symbols = df[sym_col].astype(str).str.upper().tolist()
        vec_cols = [c for c in df.columns if c != sym_col]
        vectors = df[vec_cols].to_numpy(dtype=np.float32)
        return cls(symbols=symbols, vectors=vectors)

    # ------------------------------------------------------------------ #
    def save_npz(self, path: str | Path) -> None:
        np.savez(
            path,
            symbols=np.asarray(self.symbols, dtype=object).astype("U64"),
            vectors=self.vectors.astype(np.float32),
        )

    # ------------------------------------------------------------------ #
    # 与词表对齐
    # ------------------------------------------------------------------ #
    def align_to_vocab(self, vocab: GeneVocab) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(V, d)`` prior 矩阵与 ``(V,)`` bool mask。

        词表中没有先验的 token（含全部 special tokens、未匹配基因）mask=False、
        prior 行用 0 填充。
        """
        V = len(vocab)
        d = self.dim
        out = np.zeros((V, d), dtype=np.float32)
        mask = np.zeros((V,), dtype=bool)
        prior_lookup = {s: i for i, s in enumerate(self.symbols)}
        for token_id, sym in enumerate(vocab.id_to_symbol):
            j = prior_lookup.get(sym.upper())
            if j is not None:
                out[token_id] = self.vectors[j]
                mask[token_id] = True
        return torch.from_numpy(out), torch.from_numpy(mask)
