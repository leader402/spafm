"""Hierarchical SpaFM：内层 per-cell 编码 + 外层 spot-as-token 编码。

动机：v0 的 spatial bias 在每个 cell 内部对所有基因 token 共享同一坐标，导致
``dist2 == 0``，spatial bias 形同虚设。本模块将每个 spot 整体作为外层一个 token，
外层注意力直接消费 spot-spot 距离，spatial bias 才真正起作用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from spafm.models.spafm import ModelConfig, SpaFMModel
from spafm.models.transformer import SpatialTransformerBlock


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class HierarchicalConfig:
    """Hierarchical SpaFM 配置。"""

    inner: ModelConfig = field(default_factory=ModelConfig)
    outer_n_layers: int = 2
    outer_n_heads: int = 4
    outer_d_ffn: int = 1024
    outer_dropout: float = 0.1
    outer_spatial_bias_enabled: bool = True
    outer_spatial_sigma: float = 1000.0  # spot 间距尺度（Visium ~100µm）

    @classmethod
    def from_yaml(cls, path: str | Path) -> HierarchicalConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        inner_raw = raw.pop("inner", {})
        return cls(inner=ModelConfig(**inner_raw), **raw)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class HierarchicalSpaFM(nn.Module):
    """两层结构：

    - inner: SpaFMModel（按 spot 编码基因序列）
    - outer: 若干 SpatialTransformerBlock（按 slice 编码 spot 序列）
    """

    def __init__(self, cfg: HierarchicalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner = SpaFMModel(cfg.inner)
        d_model = cfg.inner.d_model
        self.outer_blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    d_model=d_model,
                    n_heads=cfg.outer_n_heads,
                    d_ffn=cfg.outer_d_ffn,
                    dropout=cfg.outer_dropout,
                    spatial_bias=cfg.outer_spatial_bias_enabled,
                    spatial_sigma=cfg.outer_spatial_sigma,
                )
                for _ in range(cfg.outer_n_layers)
            ]
        )
        self.outer_norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls, cfg: HierarchicalConfig | str | Path
    ) -> HierarchicalSpaFM:
        if isinstance(cfg, (str, Path)):
            cfg = HierarchicalConfig.from_yaml(cfg)
        return cls(cfg)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        gene_ids: torch.Tensor,  # (B, N, L)
        pos_emb: torch.Tensor,  # (B, N, L, d_pos)
        attention_mask: torch.Tensor,  # (B, N, L) bool
        spot_coords: torch.Tensor,  # (B, N, 2)
        spot_attention_mask: torch.Tensor,  # (B, N) bool
        coords: torch.Tensor | None = None,  # (B, N, L, 2) — 内层 spatial bias 用
        value_ids: torch.Tensor | None = None,  # (B, N, L)
        value_floats: torch.Tensor | None = None,  # (B, N, L)
        return_gene_logits: bool = False,
        return_attn: bool = False,
        return_inner_attn: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        B, N, L = gene_ids.shape

        # ---- 内层：把 (B, N) 展平为 cells 维 ---------------------------- #
        flat_gene_ids = gene_ids.reshape(B * N, L)
        flat_mask = attention_mask.reshape(B * N, L)
        d_pos = pos_emb.shape[-1]
        flat_pos = pos_emb.reshape(B * N, L, d_pos)
        flat_coords = (
            coords.reshape(B * N, L, 2) if coords is not None else None
        )
        flat_value_ids = value_ids.reshape(B * N, L) if value_ids is not None else None
        flat_value_floats = (
            value_floats.reshape(B * N, L) if value_floats is not None else None
        )

        inner_out = self.inner(
            gene_ids=flat_gene_ids,
            pos_emb=flat_pos,
            attention_mask=flat_mask,
            coords=flat_coords,
            value_ids=flat_value_ids,
            value_floats=flat_value_floats,
            return_gene_logits=return_gene_logits,
            return_attn=return_inner_attn,
        )
        cell_repr_flat = inner_out["cell_repr"]  # (B*N, d)
        d = cell_repr_flat.shape[-1]
        spot_tokens = cell_repr_flat.reshape(B, N, d)

        # ---- 外层：spot-as-token 注意力 -------------------------------- #
        x = spot_tokens
        outer_attns: list[torch.Tensor] = []
        for blk in self.outer_blocks:
            if return_attn:
                x, attn = blk(
                    x,
                    attention_mask=spot_attention_mask,
                    coords=spot_coords,
                    return_attn=True,
                )
                outer_attns.append(attn)
            else:
                x = blk(x, attention_mask=spot_attention_mask, coords=spot_coords)
        x = self.outer_norm(x)
        spot_repr = x  # (B, N, d)

        out: dict[str, torch.Tensor] = {
            "token_repr": inner_out["token_repr"].reshape(B, N, L, d),
            "cell_repr": spot_tokens,  # 内层 CLS
            "spot_repr": spot_repr,  # 外层精炼后的 spot 表示
        }
        if return_gene_logits and "gene_logits" in inner_out:
            V = inner_out["gene_logits"].shape[-1]
            out["gene_logits"] = inner_out["gene_logits"].reshape(B, N, L, V)
        if return_attn:
            out["outer_attentions"] = outer_attns  # type: ignore[assignment]
        if return_inner_attn and "attentions" in inner_out:
            # 每层形状 (B*N, H, L, L) → reshape到 (B, N, H, L, L)
            ia = inner_out["attentions"]
            out["inner_attentions"] = [a.reshape(B, N, *a.shape[1:]) for a in ia]  # type: ignore[assignment]
        return out
