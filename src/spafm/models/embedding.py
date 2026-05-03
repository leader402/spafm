"""SpaFM 输入嵌入层：gene + value + spatial position。"""

from __future__ import annotations

import torch
from torch import nn


class GeneEmbedding(nn.Module):
    """基因 token 嵌入；padding_idx=0（[PAD]）。"""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.emb.weight[padding_idx].zero_()

    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(gene_ids)


class ValueEmbedding(nn.Module):
    """表达量嵌入；``mode='bin'`` 用 Embedding，``'continuous'`` 用 MLP。"""

    def __init__(self, mode: str, d_model: int, n_bins: int = 51) -> None:
        super().__init__()
        if mode not in {"bin", "continuous"}:
            raise ValueError(f"mode 必须是 bin/continuous，得到 {mode}")
        self.mode = mode
        if mode == "bin":
            self.bin_emb = nn.Embedding(n_bins, d_model)
            nn.init.normal_(self.bin_emb.weight, mean=0.0, std=0.02)
        else:
            self.value_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

    def forward(
        self, value_ids: torch.Tensor | None, value_floats: torch.Tensor | None
    ) -> torch.Tensor:
        if self.mode == "bin":
            if value_ids is None:
                raise ValueError("bin 模式需要 value_ids")
            return self.bin_emb(value_ids)
        if value_floats is None:
            raise ValueError("continuous 模式需要 value_floats")
        return self.value_proj(value_floats.unsqueeze(-1))


class InputComposer(nn.Module):
    """把 gene / value / spatial 三类 embedding 加和并 LayerNorm。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_pos: int,
        expression_mode: str = "bin",
        n_value_bins: int = 51,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gene = GeneEmbedding(vocab_size, d_model)
        self.value = ValueEmbedding(expression_mode, d_model, n_bins=n_value_bins)
        self.pos_proj = nn.Linear(d_pos, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        gene_ids: torch.Tensor,
        pos_emb: torch.Tensor,
        value_ids: torch.Tensor | None = None,
        value_floats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.gene(gene_ids)
        x = x + self.value(value_ids, value_floats)
        x = x + self.pos_proj(pos_emb)
        x = self.norm(x)
        return self.dropout(x)
