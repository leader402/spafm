"""mgm_collate：把 :class:`H5ADCorpusDataset` 输出的单 cell 字典列表
合成 batch 张量字典，与 :meth:`STTokenizer.encode` 输出 schema 兼容。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from spafm.tokenization import STTokenizer
from spafm.tokenization.gene_vocab import PAD_ID


def make_collator(tokenizer: STTokenizer):
    """构造一个 closure 形式的 collate_fn。"""
    expr_mode = tokenizer.cfg.expression.get("mode", "bin")

    def _collate(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        B = len(batch)
        L = max(int(item["gene_ids"].shape[0]) for item in batch)

        gene_ids = np.full((B, L), PAD_ID, dtype=np.int64)
        coords = np.zeros((B, L, 2), dtype=np.float32)
        attn = np.zeros((B, L), dtype=bool)
        if expr_mode == "bin":
            values = np.zeros((B, L), dtype=np.int64)
        else:
            values = np.zeros((B, L), dtype=np.float32)

        for b, item in enumerate(batch):
            n = int(item["gene_ids"].shape[0])
            gene_ids[b, :n] = item["gene_ids"]
            coords[b, :n] = item["coords"]
            attn[b, :n] = True
            if expr_mode == "bin":
                values[b, :n] = item["values"].astype(np.int64, copy=False)
            else:
                values[b, :n] = item["values"].astype(np.float32, copy=False)

        # 统一在 collator 里算 pos_emb（避免 worker 额外算两次空间编码）
        pos_emb_flat = tokenizer._encode_pos(coords.reshape(-1, 2))
        pos_emb = pos_emb_flat.reshape(B, L, -1).astype(np.float32, copy=False)

        out: dict[str, torch.Tensor] = {
            "gene_ids": torch.from_numpy(gene_ids),
            "coords": torch.from_numpy(coords),
            "pos_emb": torch.from_numpy(pos_emb),
            "attention_mask": torch.from_numpy(attn),
        }
        if expr_mode == "bin":
            out["value_ids"] = torch.from_numpy(values)
        else:
            out["value_floats"] = torch.from_numpy(values)
        return out

    return _collate


def mgm_collate(batch: list[dict[str, np.ndarray]], tokenizer: STTokenizer) -> dict[str, Any]:
    """便捷函数（当显式提供 tokenizer 时使用）。"""
    return make_collator(tokenizer)(batch)
