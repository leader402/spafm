"""空间位置编码：sin-cos 2D 与 Random Fourier Features。"""

from __future__ import annotations

import numpy as np


def sincos2d(coords: np.ndarray, dim: int = 128, max_period: float = 10000.0) -> np.ndarray:
    """对 (N, 2) 坐标做 sin-cos 位置编码。

    Args:
        coords: 形状 ``(N, 2)``，已经归一化到 [0, 1] 量级最佳。
        dim: 输出维度，必须是 4 的倍数（每轴 dim/2，再分 sin/cos）。
        max_period: 最长周期。

    Returns:
        ``(N, dim)`` float32。
    """
    if dim % 4 != 0:
        raise ValueError(f"dim 必须是 4 的倍数，得到 {dim}")
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords 形状必须为 (N, 2)，得到 {coords.shape}")

    n, _ = coords.shape
    half = dim // 2  # 每个轴
    freqs = np.exp(
        -np.log(max_period) * np.arange(0, half, 2, dtype=np.float32) / half
    )  # (half/2,)
    out = np.zeros((n, dim), dtype=np.float32)
    for axis in (0, 1):
        ang = coords[:, axis : axis + 1] * freqs[None, :]  # (N, half/2)
        offset = axis * half
        out[:, offset : offset + half // 2] = np.sin(ang)
        out[:, offset + half // 2 : offset + half] = np.cos(ang)
    return out


def rff2d(coords: np.ndarray, dim: int = 128, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """Random Fourier Features for 2D。

    Args:
        coords: ``(N, 2)``。
        dim: 输出维度，必须是偶数。
        sigma: Gaussian 频率方差。
        seed: 随机种子（保证不同调用一致）。
    """
    if dim % 2 != 0:
        raise ValueError(f"dim 必须是偶数，得到 {dim}")
    coords = np.asarray(coords, dtype=np.float32)
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, sigma, size=(2, dim // 2)).astype(np.float32)
    proj = 2.0 * np.pi * coords @ B  # (N, dim/2)
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=1)
