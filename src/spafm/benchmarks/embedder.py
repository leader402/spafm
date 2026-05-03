"""SpaFMEmbedder：把 SpaFMModel 包成"输入 h5ad → 输出 (cell_repr, token_repr+mask)"。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from spafm.models import ModelConfig, SpaFMModel
from spafm.tokenization import STTokenizer
from spafm.training import H5ADCorpusDataset
from spafm.training.collator import make_collator


class SpaFMEmbedder:
    """把 :class:`SpaFMModel` 包装为评测用 embedder。"""

    name = "spafm"

    def __init__(
        self,
        model_config: str | Path | dict | ModelConfig,
        tokenizer: STTokenizer,
        ckpt: str | Path | None = None,
        batch_size: int = 8,
        device: str = "cpu",
    ) -> None:
        if isinstance(model_config, (str, Path)):
            mc = ModelConfig.from_yaml(model_config)
        elif isinstance(model_config, dict):
            mc = ModelConfig(**model_config)
        else:
            mc = model_config
        if len(tokenizer.vocab) > mc.vocab_size:
            mc.vocab_size = len(tokenizer.vocab)
        self.model_cfg = mc
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

        self.model = SpaFMModel(mc).to(self.device).eval()
        if ckpt:
            self._load_ckpt(ckpt)

    def _load_ckpt(self, path: str | Path) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        bb = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
        if not bb:
            bb = sd  # 兼容直接保存 model.state_dict()
        missing, unexpected = self.model.load_state_dict(bb, strict=False)
        print(f"[SpaFMEmbedder] ckpt missing={len(missing)} unexpected={len(unexpected)}")

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def embed(self, files: list[str | Path]) -> dict[str, Any]:
        """前向整个 dataset，返回 cell-level + token-level 表征。

        Returns
        -------
        dict 含:
            ``cell_repr``  shape (N, d_model)
            ``token_repr`` list[ndarray (L_i, d_model)]
            ``gene_ids``   list[ndarray (L_i,)]
            ``values``     list[ndarray (L_i,)]  原始（bin id 或 float）
        """
        ds = H5ADCorpusDataset(files=list(files), tokenizer=self.tokenizer)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=make_collator(self.tokenizer),
        )

        cell_reprs: list[np.ndarray] = []
        token_reprs: list[np.ndarray] = []
        gene_ids_all: list[np.ndarray] = []
        values_all: list[np.ndarray] = []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(
                gene_ids=batch["gene_ids"],
                pos_emb=batch["pos_emb"],
                attention_mask=batch["attention_mask"],
                coords=batch.get("coords"),
                value_ids=batch.get("value_ids"),
                value_floats=batch.get("value_floats"),
                return_gene_logits=False,
            )
            cell = out["cell_repr"].cpu().numpy()
            tok = out["token_repr"].cpu().numpy()
            mask = batch["attention_mask"].cpu().numpy()
            gid = batch["gene_ids"].cpu().numpy()
            if "value_floats" in batch:
                val = batch["value_floats"].cpu().numpy()
            else:
                val = batch["value_ids"].cpu().numpy()

            for b in range(cell.shape[0]):
                m = mask[b]
                cell_reprs.append(cell[b])
                token_reprs.append(tok[b][m])
                gene_ids_all.append(gid[b][m])
                values_all.append(val[b][m])

        return {
            "cell_repr": np.stack(cell_reprs, axis=0),
            "token_repr": token_reprs,
            "gene_ids": gene_ids_all,
            "values": values_all,
        }


# --------------------------------------------------------------------------- #
class HierSpaFMEmbedder:
    """把 :class:`HierarchicalSpaFM` 包装为 spot-level embedder。

    与 :class:`SpaFMEmbedder` 的差异：
    - 加载分层 ckpt（外层 transformer 也参与）；
    - ``cell_repr`` 取 ``out['spot_repr']``（外层精炼后的空间感知表征）；
    - 按 slice 顺序遍历，每 slice 内分块前向，**保持原 obs 顺序**输出。

    适用于：层级聚类 / cell type linear probe 等下游任务。
    """

    name = "spafm_hier"

    def __init__(
        self,
        model_config: str | Path | dict,
        tokenizer: STTokenizer,
        ckpt: str | Path | None = None,
        spots_per_batch: int = 64,
        device: str = "cpu",
    ) -> None:
        from spafm.benchmarks.ccc import load_hier_from_ckpt
        from spafm.models.hierarchical import HierarchicalConfig, HierarchicalSpaFM

        self.tokenizer = tokenizer
        self.device = device
        self.spots_per_batch = int(spots_per_batch)
        if ckpt is not None:
            self.model = load_hier_from_ckpt(
                ckpt_path=ckpt,
                model_config=model_config,
                vocab_size_override=len(tokenizer.vocab),
                device=device,
            )
        else:
            if isinstance(model_config, (str, Path)):
                cfg = HierarchicalConfig.from_yaml(model_config)
            elif isinstance(model_config, dict):
                from spafm.models.spafm import ModelConfig

                inner = model_config.get("inner", {})
                rest = {k: v for k, v in model_config.items() if k != "inner"}
                cfg = HierarchicalConfig(inner=ModelConfig(**inner), **rest)
            else:
                cfg = model_config
            cfg.inner.vocab_size = max(int(cfg.inner.vocab_size), len(tokenizer.vocab))
            self.model = HierarchicalSpaFM(cfg).to(device).eval()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def embed(self, files: list[str | Path]) -> dict[str, Any]:
        """对每个 slice 顺序前向所有 spot，输出 ``cell_repr=(N_total, d)``。"""
        from anndata import read_h5ad

        from spafm.benchmarks.ccc import _row
        from spafm.training.slice_dataset import make_slice_collator

        cell_reprs: list[np.ndarray] = []
        token_reprs: list[np.ndarray] = []
        gene_ids_all: list[np.ndarray] = []
        values_all: list[np.ndarray] = []

        for f in files:
            adata = read_h5ad(f)
            n = adata.n_obs
            var_token_ids = self.tokenizer._gene_id_array(adata)
            coords_full = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]

            for s_start in range(0, n, self.spots_per_batch):
                s_end = min(n, s_start + self.spots_per_batch)
                n_chunk = s_end - s_start
                spot_dicts = []
                for ri in range(s_start, s_end):
                    row = _row(adata, int(ri))
                    spot_dicts.append(
                        self.tokenizer.encode_one(
                            row_counts=row,
                            coord=coords_full[int(ri)],
                            var_token_ids=var_token_ids,
                        )
                    )
                item = {
                    "spot_dicts": spot_dicts,
                    "spot_coords": coords_full[s_start:s_end],
                    "spot_attention_mask": np.ones(n_chunk, dtype=bool),
                    "n_spots_valid": np.int64(n_chunk),
                    "slice_idx": np.int64(0),
                }
                coll = make_slice_collator(self.tokenizer, n_spots_per_sample=n_chunk)
                batch = coll([item])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(
                    gene_ids=batch["gene_ids"],
                    pos_emb=batch["pos_emb"],
                    attention_mask=batch["attention_mask"],
                    spot_coords=batch["spot_coords"],
                    spot_attention_mask=batch["spot_attention_mask"],
                    coords=batch["coords"],
                    value_ids=batch.get("value_ids"),
                    value_floats=batch.get("value_floats"),
                )
                # spot_repr: (1, n_chunk, d)
                spot_rep = out["spot_repr"][0].cpu().numpy()
                tok = out["token_repr"][0].cpu().numpy()  # (n_chunk, L, d)
                mask = batch["attention_mask"][0].cpu().numpy()
                gid = batch["gene_ids"][0].cpu().numpy()
                val = (
                    batch["value_floats"][0].cpu().numpy()
                    if "value_floats" in batch
                    else batch["value_ids"][0].cpu().numpy()
                )
                for j in range(n_chunk):
                    m = mask[j]
                    cell_reprs.append(spot_rep[j])
                    token_reprs.append(tok[j][m])
                    gene_ids_all.append(gid[j][m])
                    values_all.append(val[j][m])

                del out, batch
                if str(self.device).startswith("cuda"):
                    torch.cuda.empty_cache()

        return {
            "cell_repr": np.stack(cell_reprs, axis=0),
            "token_repr": token_reprs,
            "gene_ids": gene_ids_all,
            "values": values_all,
        }
