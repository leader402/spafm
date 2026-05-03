"""LoRA：低秩适配 nn.Linear。

公式: ``y = W x + (alpha / r) * B (A x)``

- ``A`` Kaiming 初始化，``B`` 全 0（保证加载即等价原层）
- 原 ``W``、``bias`` 冻结
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """包装一个已存在的 :class:`nn.Linear`。"""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r 必须为正整数")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 保持 0 → 初始增量为 0

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scaling


# --------------------------------------------------------------------------- #
def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: tuple[str, ...] = ("qkv", "out"),
) -> nn.Module:
    """递归把 ``model`` 中所有名字结尾匹配 ``target_modules`` 的 ``nn.Linear``
    替换为 :class:`LoRALinear`（原地修改）。返回同一个 ``model``。

    匹配按 ``module 名字最后一段``，例如 ``model.blocks.0.attn.qkv``
    最后一段 ``qkv`` ∈ target_modules 就替换。
    """
    for _parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_modules:
                wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, wrapped)
    return model


def mark_only_lora_as_trainable(model: nn.Module, train_head_prefix: tuple[str, ...] = ()) -> None:
    """冻结所有非 LoRA 参数；可选放开以 ``train_head_prefix`` 开头的参数（如下游 head）。"""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
            continue
        if any(name.startswith(pref) for pref in train_head_prefix):
            p.requires_grad = True
            continue
        p.requires_grad = False


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
