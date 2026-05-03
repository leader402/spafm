"""Masked Gene Modeling 的 mask 工具。

参考 BERT MLM 80/10/10 策略：
- 在 ``attention_mask=True`` 且非特殊 token 的位置中随机选 ``mask_ratio`` 比例
- 80% 替换为 [MASK]，10% 替换为随机 gene id，10% 保持原样
- 返回掩码后的 ``gene_ids``、原始 target、boolean mask 矩阵
"""

from __future__ import annotations

import torch

from spafm.tokenization.gene_vocab import (
    BOS_ID,
    CLS_ID,
    EOS_ID,
    MASK_ID,
    NICHE_ID,
    PAD_ID,
    SEP_ID,
    UNK_ID,
)

_SPECIAL_IDS = {PAD_ID, CLS_ID, MASK_ID, UNK_ID, SEP_ID, BOS_ID, EOS_ID, NICHE_ID}


def _is_maskable(gene_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """返回 (B, L) bool：True=该位置可被 mask。"""
    not_special = torch.ones_like(gene_ids, dtype=torch.bool)
    for sid in _SPECIAL_IDS:
        not_special &= gene_ids != sid
    return attention_mask & not_special


def apply_mgm_mask(
    gene_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vocab_size: int,
    mask_ratio: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 ``gene_ids`` 做 MLM-style mask。

    Returns:
        masked_gene_ids: 与输入同形状，部分位置被替换。
        mask_positions: ``(B, L)`` bool，True 即被选中位置（用于损失计算）。
    """
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio 必须在 (0, 1) 之间")
    device = gene_ids.device
    maskable = _is_maskable(gene_ids, attention_mask)

    rand = torch.rand(gene_ids.shape, device=device, generator=generator)
    mask_positions = (rand < mask_ratio) & maskable

    masked = gene_ids.clone()
    # 在已选中的位置上再决定具体替换策略
    sub = torch.rand(gene_ids.shape, device=device, generator=generator)
    to_mask_token = mask_positions & (sub < mask_token_prob)
    to_random = (
        mask_positions & (sub >= mask_token_prob) & (sub < mask_token_prob + random_token_prob)
    )
    # 其余 mask_positions 保持原样

    masked = torch.where(to_mask_token, torch.full_like(masked, MASK_ID), masked)
    if to_random.any():
        # 随机 id 范围避开特殊 token：[len(specials), vocab_size)
        low = max(_SPECIAL_IDS) + 1
        rand_ids = torch.randint(
            low=low,
            high=vocab_size,
            size=gene_ids.shape,
            device=device,
            generator=generator,
        )
        masked = torch.where(to_random, rand_ids, masked)

    return masked, mask_positions
