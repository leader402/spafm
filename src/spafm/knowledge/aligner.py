"""PriorAligner：把 GeneEmbedding 投影到先验空间并计算 cosine 对齐 loss。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PriorAligner(nn.Module):
    """投影 + 缓存先验矩阵。

    Args:
        d_model:  GeneEmbedding 维度。
        d_prior:  外部先验维度。
        prior_matrix:  ``(V, d_prior)`` 张量，会作为 buffer 注册（不参与梯度）。
        prior_mask:    ``(V,)`` bool buffer。
        freeze_prior:  当前 v0 始终冻结先验（buffer 不入 optimizer）。
    """

    def __init__(
        self,
        d_model: int,
        d_prior: int,
        prior_matrix: torch.Tensor,
        prior_mask: torch.Tensor,
        freeze_prior: bool = True,
    ) -> None:
        super().__init__()
        if prior_matrix.dim() != 2 or prior_matrix.shape[1] != d_prior:
            raise ValueError(f"prior_matrix 形状应为 (V, {d_prior})")
        if prior_mask.shape[0] != prior_matrix.shape[0]:
            raise ValueError("prior_matrix 与 prior_mask 行数不一致")
        self.proj = nn.Linear(d_model, d_prior, bias=False)
        self.register_buffer("prior_matrix", prior_matrix.float())
        self.register_buffer("prior_mask", prior_mask.bool())
        self._freeze = bool(freeze_prior)

    def forward(self, gene_embedding_weight: torch.Tensor) -> torch.Tensor:
        """计算 alignment loss。``gene_embedding_weight`` 形状 ``(V, d_model)``。"""
        return alignment_loss(
            embedding_weight=gene_embedding_weight,
            prior_matrix=self.prior_matrix,
            prior_mask=self.prior_mask,
            projection=self.proj,
        )


def alignment_loss(
    embedding_weight: torch.Tensor,
    prior_matrix: torch.Tensor,
    prior_mask: torch.Tensor,
    projection: nn.Linear | None = None,
) -> torch.Tensor:
    """``1 - cos(E_g·W, P_g)`` 在 mask=True 行上的均值。

    若 ``projection`` 为空，则要求 ``embedding_weight`` 与 ``prior_matrix``
    维度一致并直接对齐。
    """
    if prior_mask.sum() == 0:
        return embedding_weight.sum() * 0.0
    if projection is not None:
        proj_e = projection(embedding_weight)  # (V, d_prior)
    else:
        if embedding_weight.shape[1] != prior_matrix.shape[1]:
            raise ValueError("无 projection 时维度必须匹配")
        proj_e = embedding_weight
    sel_e = proj_e[prior_mask]
    sel_p = prior_matrix[prior_mask]
    cos = F.cosine_similarity(sel_e, sel_p, dim=-1)
    return (1.0 - cos).mean()
