"""SpaFM 扩展规律层 —— Stage 9。"""

from __future__ import annotations

from spafm.scaling.flops import estimate_flops_per_token, estimate_total_flops
from spafm.scaling.params import count_params, estimate_params_from_cfg, measured_params
from spafm.scaling.scaling_law import ScalingLawFit, fit_scaling_law
from spafm.scaling.sizes import SIZE_CONFIGS, get_size_config

__all__ = [
    "SIZE_CONFIGS",
    "ScalingLawFit",
    "count_params",
    "estimate_flops_per_token",
    "estimate_params_from_cfg",
    "estimate_total_flops",
    "fit_scaling_law",
    "get_size_config",
    "measured_params",
]
