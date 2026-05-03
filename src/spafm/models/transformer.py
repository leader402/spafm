"""SpatialTransformerBlock：Pre-LN + MHSA(spatial bias) + GEGLU FFN。"""

from __future__ import annotations

import torch
from torch import nn

from spafm.models.attention import MultiHeadSelfAttention


class GEGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_hidden * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * torch.nn.functional.gelu(b)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.geglu = GEGLU(d_model, d_ffn)
        self.proj = nn.Linear(d_ffn, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.geglu(x)))


class SpatialTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
        spatial_bias: bool = True,
        spatial_sigma: float = 200.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            spatial_bias=spatial_bias,
            spatial_sigma=spatial_sigma,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ffn, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        coords: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attn:
            attn_out, attn = self.attn(
                self.norm1(x), attention_mask=attention_mask, coords=coords, return_attn=True
            )
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, attn
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask, coords=coords)
        x = x + self.ffn(self.norm2(x))
        return x
