"""参数估计：粗略闭式 + 精确实测。"""

from __future__ import annotations

import torch
from torch import nn

from spafm.models import ModelConfig


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """精确数模型参数。"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_params_from_cfg(cfg: ModelConfig) -> dict[str, int]:
    """闭式估计 SpaFMModel 参数（忽略 LayerNorm/bias 等小项）。

    返回各部分明细 + total。
    """
    V = int(cfg.vocab_size)
    d = int(cfg.d_model)
    L = int(cfg.n_layers)
    df = int(cfg.d_ffn)
    n_bins = int(cfg.n_value_bins)
    d_pos = int(cfg.d_pos)

    embed = V * d  # gene embedding
    if not cfg.tie_gene_embedding:
        embed += V * d  # 输出投影
    embed += n_bins * d  # value embedding（bin 模式）
    embed += d_pos * d  # 位置投影到 d_model

    # 每层 attention：qkv (d×3d) + out (d×d) ≈ 4 d^2
    # GEGLU FFN：通常 2*d*df + d*df = 3 d df
    per_layer = 4 * d * d + 3 * d * df
    blocks = L * per_layer

    total = embed + blocks
    return {
        "embedding": embed,
        "blocks": blocks,
        "total": total,
    }


@torch.no_grad()
def measured_params(cfg: ModelConfig) -> int:
    """构造一份最小 SpaFMModel 实例并精确数参数。"""
    from spafm.models import SpaFMModel

    return count_params(SpaFMModel(cfg))
