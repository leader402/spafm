"""scaling-law 拟合：``L(P) = A / P^alpha``（v0 简化版，无常数项 E）。

输入 ``[(P_i, L_i)]``，对 ``log L = log A - alpha log P`` 做最小二乘。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalingLawFit:
    alpha: float  # 指数
    A: float  # 系数（exp(intercept)）
    r2: float  # 决定系数

    def predict(self, P: float | np.ndarray) -> np.ndarray:
        return self.A * np.power(np.asarray(P, dtype=np.float64), -self.alpha)


def fit_scaling_law(points: list[tuple[float, float]]) -> ScalingLawFit:
    """返回 ``ScalingLawFit``。``points`` 为 ``[(params, loss), ...]``。"""
    if len(points) < 2:
        raise ValueError("至少需要 2 个 (params, loss) 数据点")
    arr = np.asarray(points, dtype=np.float64)
    P, Lv = arr[:, 0], arr[:, 1]
    if (P <= 0).any() or (Lv <= 0).any():
        raise ValueError("params/loss 必须为正")
    x = np.log(P)
    y = np.log(Lv)
    # y = b - alpha * x
    slope, intercept = np.polyfit(x, y, 1)
    alpha = float(-slope)
    A = float(np.exp(intercept))
    # R^2
    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return ScalingLawFit(alpha=alpha, A=A, r2=r2)
