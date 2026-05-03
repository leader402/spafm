"""SpaFMPretrainModule：Lightning 训练模块。

组装：

- :class:`SpaFMModel` 主干
- :class:`ContrastiveHead` 用于 cell-level 对比
- AdamW + 线性 warmup → cosine 退火
- Train step：MGM + 双 view InfoNCE
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

from spafm.models import ContrastiveHead, ModelConfig, SpaFMModel
from spafm.training.losses import info_nce, mgm_loss
from spafm.training.masking import apply_mgm_mask


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class PretrainConfig:
    """SpaFMPretrainModule 配置。"""

    model_config: str | dict | None = None  # 路径或内联 dict
    masking: dict[str, Any] = field(
        default_factory=lambda: {
            "mask_ratio": 0.15,
            "mask_token_prob": 0.8,
            "random_token_prob": 0.1,
        }
    )
    losses: dict[str, Any] = field(
        default_factory=lambda: {"mgm_weight": 1.0, "ccl_weight": 0.1, "ccl_temperature": 0.07}
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
    knowledge: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "prior_path": None,
            "prior_format": "npz",
            "alignment_weight": 0.1,
            "freeze_prior": True,
        }
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PretrainConfig:
        # 仅取我们关心的字段，其他（data/trainer/batch_size）由训练脚本处理
        keys = {"model_config", "masking", "losses", "optim", "knowledge"}
        return cls(**{k: v for k, v in d.items() if k in keys})


# --------------------------------------------------------------------------- #
# Lightning Module
# --------------------------------------------------------------------------- #
class SpaFMPretrainModule(pl.LightningModule):
    """SpaFM 自监督预训练模块。"""

    def __init__(self, cfg: PretrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # 解析 model_config
        mc = cfg.model_config
        if isinstance(mc, (str, Path)):
            self.model_cfg = ModelConfig.from_yaml(mc)
        elif isinstance(mc, dict):
            self.model_cfg = ModelConfig(**mc)
        elif isinstance(mc, ModelConfig):
            self.model_cfg = mc
        else:
            self.model_cfg = ModelConfig()

        self.model = SpaFMModel(self.model_cfg)
        self.contrastive = ContrastiveHead(d_model=self.model_cfg.d_model, d_proj=128)

        # 可选：知识对齐（Stage 5）。需要外部调用 attach_prior_aligner() 挂载，
        # 避免 Module 脱离 tokenizer 词表独立初始化。
        self.prior_aligner = None

        # save_hyperparameters 仅记录可序列化的字段（避免在 ckpt 中存 dataclass 对象，
        # PyTorch 2.6+ 默认 weights_only=True 会拒绝未注册的全局类型）
        cfg_for_hp = {k: v for k, v in cfg.__dict__.items() if k != "model_config"}
        self.save_hyperparameters({"cfg": cfg_for_hp, "model_cfg": self.model_cfg.__dict__})

    # ------------------------------------------------------------------ #
    def attach_prior_aligner(self, aligner) -> None:
        """挂载外部构造好的 :class:`PriorAligner`。"""
        from spafm.knowledge import PriorAligner  # 延迟导入避免环依赖

        if not isinstance(aligner, PriorAligner):
            raise TypeError("需要 spafm.knowledge.PriorAligner")
        self.prior_aligner = aligner

    # ------------------------------------------------------------------ #
    # 损失计算（供 training_step / validation_step 复用）
    # ------------------------------------------------------------------ #
    def _compute_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        gene_ids = batch["gene_ids"]
        attn = batch["attention_mask"]

        m_cfg = self.cfg.masking
        masked_ids, mask_positions = apply_mgm_mask(
            gene_ids,
            attn,
            vocab_size=self.model_cfg.vocab_size,
            mask_ratio=float(m_cfg["mask_ratio"]),
            mask_token_prob=float(m_cfg["mask_token_prob"]),
            random_token_prob=float(m_cfg["random_token_prob"]),
        )

        # view-1：mask 输入
        forward_kwargs = dict(
            gene_ids=masked_ids,
            pos_emb=batch["pos_emb"],
            attention_mask=attn,
            coords=batch.get("coords"),
            value_ids=batch.get("value_ids"),
            value_floats=batch.get("value_floats"),
            return_gene_logits=True,
        )
        out1 = self.model(**forward_kwargs)
        l_mgm = mgm_loss(out1["gene_logits"], gene_ids, mask_positions)

        # view-2：原始输入（不 mask）作为对比 view
        forward_kwargs2 = dict(forward_kwargs)
        forward_kwargs2["gene_ids"] = gene_ids
        forward_kwargs2["return_gene_logits"] = False
        out2 = self.model(**forward_kwargs2)

        z1 = self.contrastive(out1["cell_repr"])
        z2 = self.contrastive(out2["cell_repr"])
        l_ccl = info_nce(z1, z2, temperature=float(self.cfg.losses["ccl_temperature"]))

        w_mgm = float(self.cfg.losses["mgm_weight"])
        w_ccl = float(self.cfg.losses["ccl_weight"])
        total = w_mgm * l_mgm + w_ccl * l_ccl

        # 可选：基因先验对齐（Stage 5）
        l_align = torch.tensor(0.0, device=total.device)
        if self.prior_aligner is not None:
            w_align = float(self.cfg.knowledge.get("alignment_weight", 0.0))
            if w_align > 0:
                l_align = self.prior_aligner(self.model.embed.gene.emb.weight)
                total = total + w_align * l_align

        return {
            "loss": total,
            "loss_mgm": l_mgm.detach(),
            "loss_ccl": l_ccl.detach(),
            "loss_align": l_align.detach(),
            "n_masked": mask_positions.sum().detach().float(),
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
                "train/loss_align": out["loss_align"],
                "train/n_masked": out["n_masked"],
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
                "val/loss_align": out["loss_align"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        return out["loss"]

    # ------------------------------------------------------------------ #
    # 优化器与调度器
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        o = self.cfg.optim
        # 区分 weight decay：bias / LayerNorm 不衰减
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


# --------------------------------------------------------------------------- #
def load_pretrain_yaml(path: str | Path) -> dict[str, Any]:
    """读取顶层训练 yaml。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# 显式导出 nn 以避免未使用 import 警告
_ = nn
