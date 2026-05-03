"""SliceDataset：以 slice（h5ad 文件）为采样单元，每个样本为该 slice 中
``n_spots_per_sample`` 个 spot 的子集。

设计：
- 一个 epoch 内，每个 slice 被采样 ``samples_per_slice`` 次（不同随机子集）。
- ``__getitem__`` 返回 dict[str, np.ndarray]，每个字段 shape 为 (N, ...)，
  其中 N == n_spots_per_sample。
- 与 :func:`spafm.training.collator.make_collator` 风格保持一致，
  pos_emb 仍延后到 :func:`make_slice_collator` 中统一计算。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from anndata import read_h5ad
from torch.utils.data import Dataset

from spafm.tokenization import STTokenizer
from spafm.tokenization.gene_vocab import PAD_ID


class SliceDataset(Dataset):
    """以 slice 为单元的数据集。

    Args:
        files: ``.h5ad`` 文件列表（每个文件视为一个 slice）。
        tokenizer: STTokenizer 实例。
        n_spots_per_sample: 每个样本采样多少 spot；若 slice 不足则全取并补 padding。
        samples_per_slice: 每个 epoch 中，每个 slice 被采样的次数。
        seed: 随机种子（worker_init 中可叠加 worker_id）。
    """

    def __init__(
        self,
        files: list[str | Path],
        tokenizer: STTokenizer,
        n_spots_per_sample: int = 64,
        samples_per_slice: int = 8,
        seed: int = 0,
    ) -> None:
        if not files:
            raise ValueError("SliceDataset 至少需要 1 个 h5ad 文件")
        self.files = [Path(f) for f in files]
        self.tokenizer = tokenizer
        self.n_spots_per_sample = int(n_spots_per_sample)
        self.samples_per_slice = int(samples_per_slice)
        self._rng = np.random.default_rng(seed)

        # 一次性加载（demo 量小）
        self._adatas = [read_h5ad(f) for f in self.files]
        self._var_token_ids = [tokenizer._gene_id_array(a) for a in self._adatas]
        self._coords = []
        for a in self._adatas:
            if "spatial" not in a.obsm:
                raise KeyError(f"{a} 缺少 obsm['spatial']")
            self._coords.append(np.asarray(a.obsm["spatial"], dtype=np.float32)[:, :2])

        # 索引：(file_idx, sample_idx_within_slice)
        self._index = [
            (fi, si)
            for fi in range(len(self._adatas))
            for si in range(self.samples_per_slice)
        ]

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------ #
    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        fi, _ = self._index[i]
        adata = self._adatas[fi]
        coords_full = self._coords[fi]
        n_obs = adata.n_obs
        n_take = min(self.n_spots_per_sample, n_obs)
        sel = self._rng.choice(n_obs, size=n_take, replace=False)

        X = adata.X
        spot_dicts = []
        for ri in sel:
            row = X[int(ri)]
            row_dense = (
                np.asarray(row.todense()).ravel() if sp.issparse(row) else np.asarray(row).ravel()
            )
            spot_dicts.append(
                self.tokenizer.encode_one(
                    row_counts=row_dense,
                    coord=coords_full[int(ri)],
                    var_token_ids=self._var_token_ids[fi],
                )
            )

        # pad 到 n_spots_per_sample
        n_pad = self.n_spots_per_sample - n_take
        spot_attention_mask = np.concatenate(
            [np.ones(n_take, dtype=bool), np.zeros(n_pad, dtype=bool)]
        )
        spot_coords = np.zeros((self.n_spots_per_sample, 2), dtype=np.float32)
        spot_coords[:n_take] = coords_full[sel]

        return {
            "spot_dicts": spot_dicts,  # 长度 n_take，留给 collator 统一对齐 L
            "spot_coords": spot_coords,
            "spot_attention_mask": spot_attention_mask,
            "n_spots_valid": np.int64(n_take),
            "slice_idx": np.int64(fi),
        }


# --------------------------------------------------------------------------- #
# slice collator
# --------------------------------------------------------------------------- #
def make_slice_collator(tokenizer: STTokenizer, n_spots_per_sample: int):
    """合成 (B, N, L) 形状的 batch 字典。"""
    expr_mode = tokenizer.cfg.expression.get("mode", "bin")
    N = int(n_spots_per_sample)

    def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        B = len(batch)
        # 计算全局 L_max
        L = 0
        for item in batch:
            for sd in item["spot_dicts"]:
                L = max(L, int(sd["gene_ids"].shape[0]))
        if L == 0:
            raise RuntimeError("空 batch")

        gene_ids = np.full((B, N, L), PAD_ID, dtype=np.int64)
        coords = np.zeros((B, N, L, 2), dtype=np.float32)
        attn = np.zeros((B, N, L), dtype=bool)
        if expr_mode == "bin":
            values = np.zeros((B, N, L), dtype=np.int64)
        else:
            values = np.zeros((B, N, L), dtype=np.float32)

        spot_coords = np.zeros((B, N, 2), dtype=np.float32)
        spot_attention_mask = np.zeros((B, N), dtype=bool)

        for b, item in enumerate(batch):
            spot_coords[b] = item["spot_coords"]
            spot_attention_mask[b] = item["spot_attention_mask"]
            for n, sd in enumerate(item["spot_dicts"]):
                k = int(sd["gene_ids"].shape[0])
                gene_ids[b, n, :k] = sd["gene_ids"]
                coords[b, n, :k] = sd["coords"]
                attn[b, n, :k] = True
                if expr_mode == "bin":
                    values[b, n, :k] = sd["values"].astype(np.int64, copy=False)
                else:
                    values[b, n, :k] = sd["values"].astype(np.float32, copy=False)

        # 一次性计算 pos_emb（reshape 进 tokenizer）
        flat_coords = coords.reshape(-1, 2)
        pos_emb_flat = tokenizer._encode_pos(flat_coords)
        d_pos = pos_emb_flat.shape[-1]
        pos_emb = pos_emb_flat.reshape(B, N, L, d_pos).astype(np.float32, copy=False)

        out: dict[str, torch.Tensor] = {
            "gene_ids": torch.from_numpy(gene_ids),
            "coords": torch.from_numpy(coords),
            "pos_emb": torch.from_numpy(pos_emb),
            "attention_mask": torch.from_numpy(attn),
            "spot_coords": torch.from_numpy(spot_coords),
            "spot_attention_mask": torch.from_numpy(spot_attention_mask),
        }
        if expr_mode == "bin":
            out["value_ids"] = torch.from_numpy(values)
        else:
            out["value_floats"] = torch.from_numpy(values)
        return out

    return _collate
