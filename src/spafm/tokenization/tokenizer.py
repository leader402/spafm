"""统一 STTokenizer：AnnData → batch dict[str, Tensor]。

核心方法：

- :meth:`STTokenizer.encode_one` —— 单个 cell/spot
- :meth:`STTokenizer.encode` —— 整个 AnnData，按 obs 组装成 padded batch

输出字段：

- ``gene_ids``        : ``(B, L)`` int64
- ``value_ids``       : ``(B, L)`` int64（bin 模式）或 ``value_floats`` float32（continuous）
- ``coords``          : ``(B, L, 2)`` float32（每个 token 的 cell 物理坐标，CLS 取质心）
- ``pos_emb``         : ``(B, L, dim)`` float32（按 spatial.mode 编码）
- ``attention_mask``  : ``(B, L)`` bool（True=valid）
- ``cell_index``      : ``(B,)`` int64（在 adata 中的行号）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import yaml
from anndata import AnnData

from spafm.tokenization.expression import bin_expression, continuous_expression
from spafm.tokenization.gene_vocab import (
    CLS_ID,
    NICHE_ID,
    PAD_ID,
    GeneVocab,
)
from spafm.tokenization.spatial_encoding import rff2d, sincos2d


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class TokenizerConfig:
    """STTokenizer 行为配置。"""

    vocab_path: str | None = None
    max_genes: int = 1024
    gene_select: str = "top_k"  # top_k | random_k | all
    expression: dict[str, Any] = field(default_factory=lambda: {"mode": "bin", "n_bins": 51})
    spatial: dict[str, Any] = field(
        default_factory=lambda: {"mode": "sincos", "dim": 128, "coord_scale": 1000.0}
    )
    add_cls: bool = True
    add_niche: bool = False
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> TokenizerConfig:
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
class STTokenizer:
    """ST 数据 → 模型张量的统一 tokenizer。"""

    def __init__(self, vocab: GeneVocab, cfg: TokenizerConfig | None = None) -> None:
        self.vocab = vocab
        self.cfg = cfg or TokenizerConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------ #
    # 工厂
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: TokenizerConfig) -> STTokenizer:
        if cfg.vocab_path and Path(cfg.vocab_path).exists():
            vocab = GeneVocab.from_tsv(cfg.vocab_path)
        else:
            # fallback：空 vocab，仅 special tokens；调用 encode 时实际基因会被映射到 [UNK]
            vocab = GeneVocab.from_symbols([])
        return cls(vocab=vocab, cfg=cfg)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _gene_id_array(self, adata: AnnData) -> np.ndarray:
        """对 adata.var 整体编码一次，得到 (n_vars,) 的 token id 数组。"""
        if "gene_symbol" in adata.var.columns:
            symbols = adata.var["gene_symbol"].astype(str).to_numpy()
        else:
            symbols = adata.var_names.astype(str).to_numpy()
        return self.vocab.encode(symbols)

    def _select_indices(self, row_counts: np.ndarray, k: int) -> np.ndarray:
        """从单行 dense counts 中按策略挑出最多 k 个非零基因索引。"""
        nz_idx = np.flatnonzero(row_counts > 0)
        if nz_idx.size == 0:
            return nz_idx
        mode = self.cfg.gene_select
        if mode == "all" or nz_idx.size <= k:
            return nz_idx
        if mode == "top_k":
            top = np.argpartition(-row_counts[nz_idx], k - 1)[:k]
            return nz_idx[top]
        if mode == "random_k":
            return self._rng.choice(nz_idx, size=k, replace=False)
        raise ValueError(f"未知 gene_select: {mode}")

    def _encode_pos(self, coords: np.ndarray) -> np.ndarray:
        """coords (M, 2) → pos_emb (M, dim)。"""
        sp_cfg = self.cfg.spatial
        scale = float(sp_cfg.get("coord_scale", 1000.0))
        norm = coords / max(scale, 1e-6)
        mode = sp_cfg.get("mode", "sincos")
        dim = int(sp_cfg.get("dim", 128))
        if mode == "sincos":
            return sincos2d(norm, dim=dim)
        if mode == "rff":
            return rff2d(norm, dim=dim, sigma=float(sp_cfg.get("sigma", 1.0)), seed=self.cfg.seed)
        raise ValueError(f"未知 spatial mode: {mode}")

    # ------------------------------------------------------------------ #
    # 单 cell 编码
    # ------------------------------------------------------------------ #
    def encode_one(
        self,
        row_counts: np.ndarray,
        coord: np.ndarray,
        var_token_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """编码单个 cell。

        Args:
            row_counts: ``(n_vars,)`` raw counts。
            coord: ``(2,)`` 该 cell 的空间坐标。
            var_token_ids: ``(n_vars,)`` adata.var 对应的全局 token id（预计算）。

        Returns:
            dict，键见模块 docstring；不含 batch 维与 padding。
        """
        k = self.cfg.max_genes - (1 if self.cfg.add_cls else 0)
        sel = self._select_indices(row_counts, k=k)
        sel_counts = row_counts[sel]
        sel_gene_ids = var_token_ids[sel]

        # value 编码
        mode = self.cfg.expression.get("mode", "bin")
        if mode == "bin":
            n_bins = int(self.cfg.expression.get("n_bins", 51))
            sel_values = bin_expression(sel_counts, n_bins=n_bins)
            value_dtype = np.int64
        elif mode == "continuous":
            sel_values = continuous_expression(sel_counts)
            value_dtype = np.float32
        else:
            raise ValueError(f"未知 expression.mode: {mode}")

        # 拼 [CLS] + genes
        if self.cfg.add_cls:
            gene_ids = np.concatenate([[CLS_ID], sel_gene_ids]).astype(np.int64)
            values = np.concatenate(
                [
                    np.zeros(1, dtype=value_dtype),
                    sel_values.astype(value_dtype, copy=False),
                ]
            )
        else:
            gene_ids = sel_gene_ids.astype(np.int64)
            values = sel_values.astype(value_dtype, copy=False)

        coord_arr = np.tile(coord.astype(np.float32), (gene_ids.size, 1))
        return {
            "gene_ids": gene_ids,
            "values": values,
            "coords": coord_arr,
        }

    # ------------------------------------------------------------------ #
    # batch 编码（整个 AnnData）
    # ------------------------------------------------------------------ #
    def encode(
        self,
        adata: AnnData,
        cell_indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """编码整个（或部分）AnnData，返回 numpy batch（B, L, ...）。

        模型层通常会再用 ``torch.from_numpy`` 转换。
        """
        if "spatial" not in adata.obsm:
            raise KeyError("adata.obsm['spatial'] 缺失，无法编码空间位置。")
        coords_all = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]

        if cell_indices is None:
            cell_indices = np.arange(adata.n_obs, dtype=np.int64)

        var_token_ids = self._gene_id_array(adata)
        L = self.cfg.max_genes
        B = cell_indices.size

        gene_ids = np.full((B, L), PAD_ID, dtype=np.int64)
        coords = np.zeros((B, L, 2), dtype=np.float32)
        attn = np.zeros((B, L), dtype=bool)

        mode = self.cfg.expression.get("mode", "bin")
        if mode == "bin":
            values = np.zeros((B, L), dtype=np.int64)
        else:
            values = np.zeros((B, L), dtype=np.float32)

        X = adata.X
        is_sparse = sp.issparse(X)
        for b, idx in enumerate(cell_indices):
            row = X[idx]
            row_dense = np.asarray(row.todense()).ravel() if is_sparse else np.asarray(row).ravel()
            one = self.encode_one(row_dense, coords_all[idx], var_token_ids)
            n = one["gene_ids"].size
            n = min(n, L)
            gene_ids[b, :n] = one["gene_ids"][:n]
            values[b, :n] = one["values"][:n]
            coords[b, :n] = one["coords"][:n]
            attn[b, :n] = True

        # 可选 niche token：在序列尾追加（这里只占位 [NICHE]，实际邻居聚合 Stage 3 再做）
        if self.cfg.add_niche:
            for b in range(B):
                last = int(attn[b].sum())
                if last < L:
                    gene_ids[b, last] = NICHE_ID
                    coords[b, last] = coords_all[cell_indices[b]]
                    attn[b, last] = True

        pos_emb = self._encode_pos(coords.reshape(-1, 2)).reshape(B, L, -1)

        out: dict[str, np.ndarray] = {
            "gene_ids": gene_ids,
            "coords": coords,
            "pos_emb": pos_emb,
            "attention_mask": attn,
            "cell_index": cell_indices.astype(np.int64),
        }
        if mode == "bin":
            out["value_ids"] = values
        else:
            out["value_floats"] = values
        return out
