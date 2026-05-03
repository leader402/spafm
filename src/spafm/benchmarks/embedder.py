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
