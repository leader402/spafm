"""Baseline embedder：PCA / HVG-mean。

不依赖 SpaFMModel，仅用 anndata + sklearn。接口与 SpaFMEmbedder.embed 对齐：
返回 ``cell_repr``。token_repr/gene_ids/values 用空 list 占位（baseline 不做插补）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from anndata import read_h5ad
from sklearn.decomposition import PCA


def _stack_X(files: list[str | Path]) -> tuple[np.ndarray, list[str]]:
    """把多个 h5ad 的 X 按行 concat（gene 取交集）。"""
    adatas = [read_h5ad(f) for f in files]
    common = sorted(set.intersection(*(set(a.var_names.astype(str)) for a in adatas)))
    if not common:
        raise ValueError("baseline 需要 h5ad 之间存在共同基因")
    Xs = []
    for a in adatas:
        sub = a[:, common].X
        Xs.append(np.asarray(sub.toarray() if hasattr(sub, "toarray") else sub, dtype=np.float32))
    return np.concatenate(Xs, axis=0), common


class PCAEmbedder:
    """log1p → PCA(n_components)。"""

    name = "pca"

    def __init__(self, n_components: int = 50, seed: int = 0) -> None:
        self.n_components = n_components
        self.seed = seed

    def embed(self, files: list[str | Path]) -> dict[str, Any]:
        X, _ = _stack_X(files)
        Xn = np.log1p(X)
        n_comp = min(self.n_components, min(Xn.shape) - 1)
        if n_comp < 2:
            return {"cell_repr": Xn, "token_repr": [], "gene_ids": [], "values": []}
        emb = PCA(n_components=n_comp, random_state=self.seed).fit_transform(Xn)
        return {
            "cell_repr": emb.astype(np.float32),
            "token_repr": [],
            "gene_ids": [],
            "values": [],
        }


class HVGMeanEmbedder:
    """top-k 高变基因均值（极弱 baseline）。"""

    name = "hvg_mean"

    def __init__(self, top_k: int = 50) -> None:
        self.top_k = top_k

    def embed(self, files: list[str | Path]) -> dict[str, Any]:
        X, _ = _stack_X(files)
        Xn = np.log1p(X)
        var = Xn.var(axis=0)
        k = min(self.top_k, Xn.shape[1])
        idx = np.argsort(-var)[:k]
        emb = Xn[:, idx]
        return {
            "cell_repr": emb.astype(np.float32),
            "token_repr": [],
            "gene_ids": [],
            "values": [],
        }
