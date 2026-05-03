"""可复现的合成基因先验，用于单测与 demo。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spafm.knowledge.gene_priors import GenePriorBank


def build_synthetic_prior(
    symbols: list[str],
    dim: int = 64,
    n_clusters: int = 8,
    seed: int = 0,
) -> GenePriorBank:
    """生成有微弱聚类结构的合成先验。

    把 symbols 哈希到 ``n_clusters`` 个组，每组共享一个中心向量 + 个体噪声。
    便于测试 "alignment_loss 在拟合后下降"。
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    vectors = np.zeros((len(symbols), dim), dtype=np.float32)
    for i, s in enumerate(symbols):
        c = abs(hash(s)) % n_clusters
        vectors[i] = centers[c] + 0.1 * rng.standard_normal(dim).astype(np.float32)
    # L2 归一化（cosine 友好）
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-6)
    return GenePriorBank(symbols=[s.upper() for s in symbols], vectors=vectors)


def write_demo_prior(
    out_path: str | Path,
    symbols: list[str],
    dim: int = 64,
    seed: int = 0,
) -> Path:
    bank = build_synthetic_prior(symbols, dim=dim, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bank.save_npz(out_path)
    return out_path
