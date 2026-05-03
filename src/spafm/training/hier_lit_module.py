"""HierarchicalSpaFMPretrainModule：分层模型的预训练 LightningModule。

与 :class:`SpaFMPretrainModule` 的差异：
- batch 形状为 (B, N, L)：B 个 slice、每 slice N 个 spot、每 spot L 个基因 token；
- 内层（MGM）在所有 spot 上做（展平 B*N）；
- 外层（CCL）使用 ``spot_repr``，对同一 slice 的两次随机采样视图做 spot-level 对比；
  v0 简化：直接对 (B*N, d) 维度做双 view InfoNCE，正样本=同一位置同一 view 的 mask/no-mask 副本。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import yaml
from torch import nn

from spafm.models import ContrastiveHead
from spafm.models.hierarchical import HierarchicalConfig, HierarchicalSpaFM
from spafm.models.spafm import ModelConfig
from spafm.training.losses import info_nce, mgm_loss
from spafm.training.masking import apply_mgm_mask


# --------------------------------------------------------------------------- #
@dataclass
class HierPretrainConfig:
    """分层模型预训练配置。"""

    model_config: str | dict | None = None  # HierarchicalConfig 的 yaml 或 dict
    masking: dict[str, Any] = field(
        default_factory=lambda: {
            "mask_ratio": 0.15,
            "mask_token_prob": 0.8,
            "random_token_prob": 0.1,
        }
    )
    losses: dict[str, Any] = field(
        default_factory=lambda: {
            "mgm_weight": 1.0,
            "ccl_weight": 0.1,
            "ccl_temperature": 0.07,
            "use_outer_repr_for_ccl": True,
        }
    )
    optim: dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 3.0e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
            "warmup_steps": 50,
            "max_steps": 1000,
        }
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HierPretrainConfig:
        keys = {"model_config", "masking", "losses", "optim"}
        return cls(**{k: v for k, v in d.items() if k in keys})


def _resolve_hier_cfg(mc: Any) -> HierarchicalConfig:
    if isinstance(mc, (str, Path)):
        return HierarchicalConfig.from_yaml(mc)
    if isinstance(mc, dict):
        inner_raw = mc.get("inner", {})
        rest = {k: v for k, v in mc.items() if k != "inner"}
        return HierarchicalConfig(inner=ModelConfig(**inner_raw), **rest)
    if isinstance(mc, HierarchicalConfig):
        return mc
    return HierarchicalConfig()


# --------------------------------------------------------------------------- #
class HierarchicalSpaFMPretrainModule(pl.LightningModule):
    """分层 SpaFM 预训练。"""

    def __init__(self, cfg: HierPretrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.hier_cfg = _resolve_hier_cfg(cfg.model_config)
        self.model = HierarchicalSpaFM(self.hier_cfg)
        self.contrastive = ContrastiveHead(d_model=self.hier_cfg.inner.d_model, d_proj=128)

        cfg_for_hp = {k: v for k, v in cfg.__dict__.items() if k != "model_config"}
        self.save_hyperparameters(
            {
                "cfg": cfg_for_hp,
                "hier_cfg": {
                    "inner": self.hier_cfg.inner.__dict__,
                    "outer_n_layers": self.hier_cfg.outer_n_layers,
                    "outer_n_heads": self.hier_cfg.outer_n_heads,
                    "outer_d_ffn": self.hier_cfg.outer_d_ffn,
                    "outer_dropout": self.hier_cfg.outer_dropout,
                    "outer_spatial_bias_enabled": self.hier_cfg.outer_spatial_bias_enabled,
                    "outer_spatial_sigma": self.hier_cfg.outer_spatial_sigma,
                },
            }
        )

    # ------------------------------------------------------------------ #
    def _compute_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        gene_ids = batch["gene_ids"]  # (B, N, L)
        attn = batch["attention_mask"]  # (B, N, L)
        B, N, L = gene_ids.shape
        V = self.hier_cfg.inner.vocab_size

        # --- mask（在 (B*N, L) 平面上做） --- #
        flat_ids = gene_ids.reshape(B * N, L)
        flat_attn = attn.reshape(B * N, L)
        m_cfg = self.cfg.masking
        # view-1：第一组独立 mask（用于 MGM 监督）
        masked_flat1, mask_pos_flat1 = apply_mgm_mask(
            flat_ids,
            flat_attn,
            vocab_size=V,
            mask_ratio=float(m_cfg["mask_ratio"]),
            mask_token_prob=float(m_cfg["mask_token_prob"]),
            random_token_prob=float(m_cfg["random_token_prob"]),
        )
        # view-2：第二组独立 mask（仅用于 CCL，让两视图都有扰动 → 防止 z1≈z2 退化）
        masked_flat2, _ = apply_mgm_mask(
            flat_ids,
            flat_attn,
            vocab_size=V,
            mask_ratio=float(m_cfg["mask_ratio"]),
            mask_token_prob=float(m_cfg["mask_token_prob"]),
            random_token_prob=float(m_cfg["random_token_prob"]),
        )
        masked_ids1 = masked_flat1.reshape(B, N, L)
        masked_ids2 = masked_flat2.reshape(B, N, L)
        mask_positions = mask_pos_flat1.reshape(B, N, L)

        common = dict(
            pos_emb=batch["pos_emb"],
            attention_mask=attn,
            spot_coords=batch["spot_coords"],
            spot_attention_mask=batch["spot_attention_mask"],
            coords=batch.get("coords"),
            value_ids=batch.get("value_ids"),
            value_floats=batch.get("value_floats"),
        )

        # view-1: mask 输入 → 训 MGM
        out1 = self.model(gene_ids=masked_ids1, return_gene_logits=True, **common)
        # gene_logits: (B, N, L, V)
        l_mgm = mgm_loss(
            out1["gene_logits"].reshape(B * N, L, V),
            gene_ids.reshape(B * N, L),
            mask_positions.reshape(B * N, L),
        )

        # view-2: 第二组独立 mask（仅 CCL 用）
        out2 = self.model(gene_ids=masked_ids2, return_gene_logits=False, **common)

        use_outer = bool(self.cfg.losses.get("use_outer_repr_for_ccl", True))
        repr1 = out1["spot_repr"] if use_outer else out1["cell_repr"]  # (B, N, d)
        repr2 = out2["spot_repr"] if use_outer else out2["cell_repr"]

        # 仅在有效 spot 上做对比
        spot_mask_flat = batch["spot_attention_mask"].reshape(B * N)
        d = repr1.shape[-1]
        z1_all = self.contrastive(repr1.reshape(B * N, d))
        z2_all = self.contrastive(repr2.reshape(B * N, d))
        z1 = z1_all[spot_mask_flat]
        z2 = z2_all[spot_mask_flat]
        if z1.shape[0] >= 2:
            l_ccl = info_nce(z1, z2, temperature=float(self.cfg.losses["ccl_temperature"]))
            # ---- CCL 诊断指标（detach，不影响梯度） ---- #
            with torch.no_grad():
                Bz = z1.shape[0]
                sim = z1 @ z2.t()  # (Bz, Bz)
                pos_sim = sim.diag().mean()
                if Bz >= 2:
                    neg_sim = (sim.sum() - sim.diag().sum()) / (Bz * (Bz - 1))
                else:
                    neg_sim = torch.tensor(0.0, device=sim.device)
                # 对齐：||z1 - z2||^2 mean（已 L2-normalize → ∈[0, 4]）
                align = (z1 - z2).pow(2).sum(dim=-1).mean()
                # 均匀度：log E_{i!=j} exp(-2 ||z1_i - z1_j||^2)
                if Bz >= 4:
                    pdist_sq = torch.cdist(z1, z1).pow(2)
                    mask = ~torch.eye(Bz, dtype=torch.bool, device=sim.device)
                    uniform = pdist_sq[mask].mul(-2.0).exp().mean().log()
                else:
                    uniform = torch.tensor(0.0, device=sim.device)
        else:
            l_ccl = torch.tensor(0.0, device=l_mgm.device)
            pos_sim = torch.tensor(0.0, device=l_mgm.device)
            neg_sim = torch.tensor(0.0, device=l_mgm.device)
            align = torch.tensor(0.0, device=l_mgm.device)
            uniform = torch.tensor(0.0, device=l_mgm.device)

        w_mgm = float(self.cfg.losses["mgm_weight"])
        w_ccl = float(self.cfg.losses["ccl_weight"])
        total = w_mgm * l_mgm + w_ccl * l_ccl
        return {
            "loss": total,
            "loss_mgm": l_mgm.detach(),
            "loss_ccl": l_ccl.detach(),
            "n_masked": mask_positions.sum().detach().float(),
            "n_spots": spot_mask_flat.sum().detach().float(),
            "ccl_pos_sim": pos_sim.detach(),
            "ccl_neg_sim": neg_sim.detach(),
            "ccl_align": align.detach(),
            "ccl_uniform": uniform.detach(),
        }

    # ------------------------------------------------------------------ #
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self._compute_losses(batch)
        bs = batch["gene_ids"].shape[0]
        self.log_dict(
            {
                "train/loss": out["loss"],
                "train/loss_mgm": out["loss_mgm"],
                "train/loss_ccl": out["loss_ccl"],
                "train/n_masked": out["n_masked"],
                "train/n_spots": out["n_spots"],
                "train/ccl_pos_sim": out["ccl_pos_sim"],
                "train/ccl_neg_sim": out["ccl_neg_sim"],
                "train/ccl_align": out["ccl_align"],
                "train/ccl_uniform": out["ccl_uniform"],
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=bs,
        )
        return out["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self._compute_losses(batch)
        bs = batch["gene_ids"].shape[0]
        self.log_dict(
            {
                "val/loss": out["loss"],
                "val/loss_mgm": out["loss_mgm"],
                "val/loss_ccl": out["loss_ccl"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        return out["loss"]

    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        o = self.cfg.optim
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or n.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [
            {"params": decay, "weight_decay": float(o["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=float(o["lr"]),
            betas=tuple(o.get("betas", (0.9, 0.95))),
        )
        warmup = int(o.get("warmup_steps", 0))
        max_steps = int(o.get("max_steps", 1000))

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return step / max(1, warmup)
            if max_steps <= warmup:
                return 1.0
            progress = (step - warmup) / max(1, max_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def load_hier_pretrain_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_ = nn
