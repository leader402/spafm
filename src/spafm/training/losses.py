"""预训练损失函数。

- :func:`mgm_loss`：被选中位置的交叉熵（target = 原 ``value_ids``，bin 模式）
- :func:`info_nce`：标准 SimCLR-style 双向 InfoNCE
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mgm_loss(
    gene_logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask_positions: torch.Tensor,
) -> torch.Tensor:
    """Masked-Gene cross-entropy（仅在 mask 位置计入损失）。

    Args:
        gene_logits: ``(B, L, V)``。
        target_ids:  ``(B, L)``，原始 gene_ids（被 mask 之前）。
        mask_positions: ``(B, L)`` bool。

    Returns:
        scalar tensor。当无 mask 位置时返回 ``0.0`` 张量（带梯度信息）。
    """
    if mask_positions.sum() == 0:
        return gene_logits.sum() * 0.0  # 保留梯度图
    B, L, V = gene_logits.shape
    flat_logits = gene_logits.reshape(B * L, V)
    flat_target = target_ids.reshape(B * L)
    flat_mask = mask_positions.reshape(B * L)
    return F.cross_entropy(flat_logits[flat_mask], flat_target[flat_mask])


def info_nce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """对称 InfoNCE。``z1``、``z2`` 形状 ``(B, d)``，需已 L2 归一化。"""
    if z1.shape != z2.shape:
        raise ValueError(f"z1/z2 形状不一致: {z1.shape} vs {z2.shape}")
    B = z1.shape[0]
    if B < 2:
        return z1.sum() * 0.0
    logits = z1 @ z2.t() / temperature  # (B, B)
    labels = torch.arange(B, device=z1.device)
    loss12 = F.cross_entropy(logits, labels)
    loss21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss12 + loss21)
