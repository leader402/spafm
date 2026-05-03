"""Gene 词表 + special tokens。

约定：special token 占据固定 ID 0–7，gene token 从 8 起编号；vocab 通过
``data/external/gene_vocab.tsv``（``token_id, symbol, species, source``）加载，
未登记基因映射为 ``[UNK]``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: 固定 special token 名列表（顺序即 ID）
SPECIAL_TOKENS: tuple[str, ...] = (
    "[PAD]",
    "[CLS]",
    "[MASK]",
    "[UNK]",
    "[SEP]",
    "[BOS]",
    "[EOS]",
    "[NICHE]",
)

PAD_ID = 0
CLS_ID = 1
MASK_ID = 2
UNK_ID = 3
SEP_ID = 4
BOS_ID = 5
EOS_ID = 6
NICHE_ID = 7
N_SPECIAL = len(SPECIAL_TOKENS)


@dataclass
class GeneVocab:
    """Gene 词表（symbol ↔ token_id）。

    通常通过 :meth:`from_tsv` 从 ``gene_vocab.tsv`` 加载；
    单元测试可用 :meth:`from_symbols` 快速构造。
    """

    symbol_to_id: dict[str, int]
    id_to_symbol: list[str]
    species_of: dict[str, str]

    # ------------------------------------------------------------------ #
    # 构造器
    # ------------------------------------------------------------------ #
    @classmethod
    def from_symbols(cls, symbols: list[str], species: str = "human") -> GeneVocab:
        """从 symbol 列表直接构造（自动加 special tokens 在前）。"""
        symbols_upper = [s.upper() for s in symbols]
        # 去重并保持顺序
        seen: set[str] = set()
        uniq: list[str] = []
        for s in symbols_upper:
            if s not in seen:
                seen.add(s)
                uniq.append(s)

        id_to_symbol: list[str] = list(SPECIAL_TOKENS) + uniq
        symbol_to_id = {s: i for i, s in enumerate(id_to_symbol)}
        species_of = {s: species for s in uniq}
        return cls(symbol_to_id=symbol_to_id, id_to_symbol=id_to_symbol, species_of=species_of)

    @classmethod
    def from_tsv(cls, path: str | Path) -> GeneVocab:
        """从 ``gene_vocab.tsv`` 加载。

        若文件中已包含 special token（``token_id`` 0–7），按文件指定的 ID；
        否则自动在前面补齐 special tokens。
        """
        df = pd.read_csv(path, sep="\t")
        # 兼容 Stage 1 build_gene_vocab.py 输出（列名 gene_symbol）以及简化 schema（symbol）
        if "symbol" not in df.columns and "gene_symbol" in df.columns:
            df = df.rename(columns={"gene_symbol": "symbol"})
        required = {"symbol", "species"}
        if not required.issubset(df.columns):
            raise KeyError(f"gene_vocab.tsv 缺少列 {required - set(df.columns)}")

        df["symbol"] = df["symbol"].astype(str).str.upper()

        # 移除 special tokens 的行（如果有），由我们统一前置
        df = df[~df["symbol"].isin(SPECIAL_TOKENS)].drop_duplicates(subset=["symbol"])
        df = df.reset_index(drop=True)

        id_to_symbol = list(SPECIAL_TOKENS) + df["symbol"].tolist()
        symbol_to_id = {s: i for i, s in enumerate(id_to_symbol)}
        species_of = dict(zip(df["symbol"].tolist(), df["species"].tolist(), strict=True))
        return cls(symbol_to_id=symbol_to_id, id_to_symbol=id_to_symbol, species_of=species_of)

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.id_to_symbol)

    @property
    def n_special(self) -> int:
        return N_SPECIAL

    @property
    def n_genes(self) -> int:
        return len(self.id_to_symbol) - N_SPECIAL

    def encode(self, symbols: list[str] | np.ndarray) -> np.ndarray:
        """symbol 列表 → token id 数组（未登记 → ``UNK_ID``）。"""
        out = np.empty(len(symbols), dtype=np.int64)
        for i, s in enumerate(symbols):
            out[i] = self.symbol_to_id.get(str(s).upper(), UNK_ID)
        return out

    def decode(self, ids: np.ndarray) -> list[str]:
        return [self.id_to_symbol[int(i)] for i in ids]
