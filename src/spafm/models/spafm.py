"""SpaFMModel：堆叠 InputComposer + N × SpatialTransformerBlock + heads。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from spafm.models.embedding import InputComposer
from spafm.models.heads import MGMHead
from spafm.models.transformer import SpatialTransformerBlock


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """SpaFMModel 配置。"""

    vocab_size: int = 64000
    n_value_bins: int = 51
    expression_mode: str = "bin"  # bin | continuous
    d_model: int = 256
    d_pos: int = 128
    n_layers: int = 6
    n_heads: int = 4
    d_ffn: int = 1024
    dropout: float = 0.1
    spatial_bias: dict[str, Any] = field(default_factory=lambda: {"enabled": True, "sigma": 200.0})
    tie_gene_embedding: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class SpaFMModel(nn.Module):
    """空间转录组基础模型骨架（v0）。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = InputComposer(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            d_pos=cfg.d_pos,
            expression_mode=cfg.expression_mode,
            n_value_bins=cfg.n_value_bins,
            dropout=cfg.dropout,
        )
        sb_enabled = bool(cfg.spatial_bias.get("enabled", True))
        sb_sigma = float(cfg.spatial_bias.get("sigma", 200.0))
        self.blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_ffn=cfg.d_ffn,
                    dropout=cfg.dropout,
                    spatial_bias=sb_enabled,
                    spatial_sigma=sb_sigma,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)

        gene_emb = self.embed.gene.emb if cfg.tie_gene_embedding else None
        self.mgm_head = MGMHead(
            d_model=cfg.d_model,
            vocab_size=cfg.vocab_size,
            gene_embedding=gene_emb,
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: ModelConfig | str | Path) -> SpaFMModel:
        if isinstance(cfg, (str, Path)):
            cfg = ModelConfig.from_yaml(cfg)
        return cls(cfg)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        gene_ids: torch.Tensor,
        pos_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        coords: torch.Tensor | None = None,
        value_ids: torch.Tensor | None = None,
        value_floats: torch.Tensor | None = None,
        return_gene_logits: bool = False,
        return_attn: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        h = self.embed(
            gene_ids=gene_ids,
            pos_emb=pos_emb,
            value_ids=value_ids,
            value_floats=value_floats,
        )
        attn_list: list[torch.Tensor] = []
        for block in self.blocks:
            if return_attn:
                h, attn = block(
                    h, attention_mask=attention_mask, coords=coords, return_attn=True
                )
                attn_list.append(attn)
            else:
                h = block(h, attention_mask=attention_mask, coords=coords)
        h = self.final_norm(h)

        out: dict[str, torch.Tensor] = {
            "token_repr": h,
            "cell_repr": h[:, 0, :],  # CLS token
        }
        if return_gene_logits:
            out["gene_logits"] = self.mgm_head(h)
        if return_attn:
            out["attentions"] = attn_list  # type: ignore[assignment]
        return out
