"""预训练 / 下游 head。

- :class:`MGMHead`：Masked Gene Modeling，输出 (B, L, vocab_size) logits；
  支持与 ``GeneEmbedding.weight`` 绑权（tied weights）。
- :class:`ContrastiveHead`：cell 表示投影到对比学习空间。
"""

from __future__ import annotations

import torch
from torch import nn


class MGMHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        gene_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        if gene_embedding is not None:
            # 用 list 包裹避免 nn.Module 自动注册为子模块/参数（防止 count 时重复计数）
            self._gene_emb_ref: list[nn.Embedding] | None = [gene_embedding]
            self.bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self._gene_emb_ref = None
            self.decoder = nn.Linear(d_model, vocab_size, bias=True)

    @property
    def tied_weight(self) -> torch.Tensor | None:
        if self._gene_emb_ref is None:
            return None
        return self._gene_emb_ref[0].weight

    def forward(self, token_repr: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj(self.norm(token_repr)))
        w = self.tied_weight
        if w is not None:
            return h @ w.t() + self.bias
        return self.decoder(h)


class ContrastiveHead(nn.Module):
    def __init__(self, d_model: int, d_proj: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_proj),
        )

    def forward(self, cell_repr: torch.Tensor) -> torch.Tensor:
        z = self.net(cell_repr)
        return torch.nn.functional.normalize(z, dim=-1)
