"""FLOPs 估计：基于 ``F ≈ 6 P`` 的 Chinchilla 经验式。"""

from __future__ import annotations

from spafm.models import ModelConfig
from spafm.scaling.params import estimate_params_from_cfg


def estimate_flops_per_token(cfg: ModelConfig) -> float:
    """单个 token 一次前向+反向的 FLOPs ≈ 6 * non-embed params。"""
    parts = estimate_params_from_cfg(cfg)
    # 通常 scaling-law 的 6*P 用 non-embed 部分
    n_eff = parts["blocks"]
    return 6.0 * float(n_eff)


def estimate_total_flops(cfg: ModelConfig, n_tokens: float) -> float:
    """整个训练所需 FLOPs：每 token FLOPs × tokens。"""
    return estimate_flops_per_token(cfg) * float(n_tokens)
