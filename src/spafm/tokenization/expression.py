"""表达量编码：离散 bin / 连续标量。"""

from __future__ import annotations

import numpy as np


def bin_expression(values: np.ndarray, n_bins: int = 51) -> np.ndarray:
    """对一行（cell 内部）非零表达量做 log1p + 分位数分箱。

    返回与 ``values`` 等长的整型数组：

    - 0：原始 count==0
    - 1..n_bins-1：非零 count 经 ``log1p`` 后按 cell 内分位数等分

    分位数实现：使用 ``np.argsort`` 的 rank 而非 ``np.quantile``，避免对称 ties。
    """
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.int64)
    nz_mask = values > 0
    if not nz_mask.any():
        return out

    nz = values[nz_mask]
    log_nz = np.log1p(nz)
    # rank 0..n-1 → bin 1..n_bins-1
    order = np.argsort(log_nz, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(log_nz))
    bins = 1 + (ranks * (n_bins - 1) // max(len(log_nz), 1))
    bins = np.clip(bins, 1, n_bins - 1)
    out[nz_mask] = bins
    return out


def continuous_expression(values: np.ndarray) -> np.ndarray:
    """连续标量编码：直接 ``log1p``，返回 float32。"""
    return np.log1p(np.asarray(values, dtype=np.float32))
