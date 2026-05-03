"""下游任务 head。"""

from __future__ import annotations

import torch
from torch import nn


class _MLPHead(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CellTypeHead(_MLPHead):
    """cell_repr (B, d) → logits (B, C)。"""

    def __init__(self, d_model: int, num_classes: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__(d_in=d_model, hidden=hidden, d_out=num_classes, dropout=dropout)


class SpatialDomainHead(_MLPHead):
    """与 CellTypeHead 同结构，仅命名区分。"""

    def __init__(self, d_model: int, num_domains: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__(d_in=d_model, hidden=hidden, d_out=num_domains, dropout=dropout)


class ImputationHead(nn.Module):
    """token_repr (B, L, d) → 预测 (B, L) 标量表达。"""

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, token_repr: torch.Tensor) -> torch.Tensor:
        return self.net(token_repr).squeeze(-1)
