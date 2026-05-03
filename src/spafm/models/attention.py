"""Multi-head self-attention，可选 spatial-distance bias。

bias 形式：``B_ij = -alpha * ||x_i - x_j||^2 / sigma^2``，加在 attention logits 上。
不同 head 共享 ``sigma``，``alpha`` 可学习，初始化为 1。
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        spatial_bias: bool = True,
        spatial_sigma: float = 200.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} 必须能被 n_heads {n_heads} 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.spatial_bias = spatial_bias
        if spatial_bias:
            # 每个 head 一个可学习 alpha
            self.alpha = nn.Parameter(torch.ones(n_heads))
            self.register_buffer("sigma2", torch.tensor(spatial_sigma**2, dtype=torch.float32))

    def _spatial_bias(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, L, 2) -> (B, 1, L, L)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, L, L, 2)
        dist2 = (diff * diff).sum(-1)  # (B, L, L)
        # 广播到 head 维：(B, H, L, L)
        bias = -self.alpha.view(1, -1, 1, 1) * dist2.unsqueeze(1) / self.sigma2
        return bias

    def forward(
        self,
        x: torch.Tensor,  # (B, L, d)
        attention_mask: torch.Tensor,  # (B, L) bool
        coords: torch.Tensor | None = None,  # (B, L, 2)
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, L, L)
        if self.spatial_bias and coords is not None:
            attn = attn + self._spatial_bias(coords)

        # mask: True=valid → 把 invalid 的 key 设为 -inf
        key_mask = (~attention_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        attn = attn.masked_fill(key_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        # 防止整行全 -inf 导致 NaN（query 全 padding）
        attn = torch.nan_to_num(attn, nan=0.0)
        attn_for_return = attn  # 返回 dropout 之前的概率，便于解释/下游
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, L, d_head)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        out = self.proj_drop(self.out(out))
        if return_attn:
            return out, attn_for_return
        return out
