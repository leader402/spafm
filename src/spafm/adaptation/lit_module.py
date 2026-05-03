"""SpaFMFinetuneModule：装配预训练 backbone + 适配策略 + 下游 head。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from spafm.adaptation.heads import CellTypeHead, ImputationHead, SpatialDomainHead
from spafm.adaptation.lora import apply_lora, count_trainable, mark_only_lora_as_trainable
from spafm.models import ModelConfig, SpaFMModel


# --------------------------------------------------------------------------- #
@dataclass
class FinetuneConfig:
    model_config: str | dict | ModelConfig | None = None
    pretrained_ckpt: str | None = None
    adaptation: dict[str, Any] = field(
        default_factory=lambda: {
            "strategy": "lora",  # linear_probe | lora | full
            "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["qkv", "out"]},
        }
    )
    head: dict[str, Any] = field(
        default_factory=lambda: {"type": "cell_type", "num_classes": 8, "hidden": 128}
    )
    optim: dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1.0e-3,
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
            "warmup_steps": 20,
            "max_steps": 200,
        }
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FinetuneConfig:
        keys = {"model_config", "pretrained_ckpt", "adaptation", "head", "optim"}
        return cls(**{k: v for k, v in d.items() if k in keys})


# --------------------------------------------------------------------------- #
class SpaFMFinetuneModule(pl.LightningModule):
    def __init__(self, cfg: FinetuneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1) backbone
        mc = cfg.model_config
        if isinstance(mc, (str, Path)):
            self.model_cfg = ModelConfig.from_yaml(mc)
        elif isinstance(mc, dict):
            self.model_cfg = ModelConfig(**mc)
        elif isinstance(mc, ModelConfig):
            self.model_cfg = mc
        else:
            self.model_cfg = ModelConfig()
        self.backbone = SpaFMModel(self.model_cfg)

        # 2) optional pretrained ckpt（仅 backbone 子集）
        if cfg.pretrained_ckpt:
            self._load_backbone_from_ckpt(cfg.pretrained_ckpt)

        # 3) head
        self.head_type = cfg.head.get("type", "cell_type")
        d = self.model_cfg.d_model
        if self.head_type == "cell_type":
            self.head = CellTypeHead(
                d_model=d,
                num_classes=int(cfg.head["num_classes"]),
                hidden=int(cfg.head.get("hidden", 128)),
            )
        elif self.head_type == "spatial_domain":
            self.head = SpatialDomainHead(
                d_model=d,
                num_domains=int(cfg.head["num_classes"]),
                hidden=int(cfg.head.get("hidden", 128)),
            )
        elif self.head_type == "imputation":
            self.head = ImputationHead(d_model=d, hidden=int(cfg.head.get("hidden", 128)))
        else:
            raise ValueError(f"未知 head.type: {self.head_type}")

        # 4) 适配策略
        strategy = cfg.adaptation.get("strategy", "lora")
        if strategy == "linear_probe":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif strategy == "lora":
            lcfg = cfg.adaptation.get("lora", {})
            apply_lora(
                self.backbone,
                r=int(lcfg.get("r", 8)),
                alpha=int(lcfg.get("alpha", 16)),
                dropout=float(lcfg.get("dropout", 0.0)),
                target_modules=tuple(lcfg.get("target_modules", ["qkv", "out"])),
            )
            mark_only_lora_as_trainable(self.backbone)
        elif strategy == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"未知 adaptation.strategy: {strategy}")

        # head 总是可训练
        for p in self.head.parameters():
            p.requires_grad = True

        n_trainable = count_trainable(self)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[Finetune] 可训练参数 {n_trainable:,} / 总 {n_total:,}（策略={strategy}）")

        cfg_for_hp = {k: v for k, v in cfg.__dict__.items() if k != "model_config"}
        self.save_hyperparameters({"cfg": cfg_for_hp, "model_cfg": self.model_cfg.__dict__})

    # ------------------------------------------------------------------ #
    def _load_backbone_from_ckpt(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        # 取出以 "model." 开头的部分，剥离前缀
        bb_sd = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
        missing, unexpected = self.backbone.load_state_dict(bb_sd, strict=False)
        print(f"[Finetune] 加载预训练 ckpt：missing={len(missing)} unexpected={len(unexpected)}")

    # ------------------------------------------------------------------ #
    # 前向 + 损失
    # ------------------------------------------------------------------ #
    def _forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.backbone(
            gene_ids=batch["gene_ids"],
            pos_emb=batch["pos_emb"],
            attention_mask=batch["attention_mask"],
            coords=batch.get("coords"),
            value_ids=batch.get("value_ids"),
            value_floats=batch.get("value_floats"),
            return_gene_logits=False,
        )
        if self.head_type in ("cell_type", "spatial_domain"):
            logits = self.head(out["cell_repr"])
            return {"logits": logits}
        # imputation
        pred = self.head(out["token_repr"])
        return {"pred": pred}

    def _loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self._forward(batch)
        if self.head_type in ("cell_type", "spatial_domain"):
            logits = out["logits"]
            target = batch["label"].long()
            loss = F.cross_entropy(logits, target)
            acc = (logits.argmax(-1) == target).float().mean()
            return {"loss": loss, "acc": acc.detach()}
        # imputation：仅在 attention_mask=True 处算 MSE，target = value_floats（若有）或 value_ids float
        pred = out["pred"]
        if "value_floats" in batch:
            target = batch["value_floats"].float()
        else:
            target = batch["value_ids"].float()
        mask = batch["attention_mask"]
        diff = (pred - target) ** 2
        loss = (diff * mask).sum() / mask.sum().clamp(min=1)
        return {"loss": loss}

    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        out = self._loss(batch)
        bs = batch["gene_ids"].shape[0]
        log = {"train/loss": out["loss"]}
        if "acc" in out:
            log["train/acc"] = out["acc"]
        self.log_dict(log, on_step=True, prog_bar=True, batch_size=bs)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._loss(batch)
        bs = batch["gene_ids"].shape[0]
        log = {"val/loss": out["loss"]}
        if "acc" in out:
            log["val/acc"] = out["acc"]
        self.log_dict(log, on_epoch=True, prog_bar=True, batch_size=bs)
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
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": float(o["weight_decay"])},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=float(o["lr"]),
            betas=tuple(o.get("betas", (0.9, 0.95))),
        )
        warmup = int(o.get("warmup_steps", 0))
        max_steps = int(o.get("max_steps", 200))

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return step / max(1, warmup)
            if max_steps <= warmup:
                return 1.0
            progress = (step - warmup) / max(1, max_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step",
            },
        }
