"""Stage 7 评估器：把 embedder 输出 + h5ad 标签 → 任务指标。"""

from __future__ import annotations

import glob as _glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from anndata import read_h5ad

from spafm.benchmarks.baselines import HVGMeanEmbedder, PCAEmbedder
from spafm.benchmarks.embedder import HierSpaFMEmbedder, SpaFMEmbedder
from spafm.benchmarks.metrics import cluster_scores, linear_probe_cv, regression_scores
from spafm.tokenization import GeneVocab, STTokenizer, TokenizerConfig


# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkConfig:
    data: dict[str, Any] = field(default_factory=dict)
    embedder: dict[str, Any] = field(default_factory=dict)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    seed: int = 42


# --------------------------------------------------------------------------- #
def _build_embedder(cfg: dict[str, Any], tokenizer: STTokenizer):
    t = cfg.get("type", "spafm")
    if t == "spafm":
        return SpaFMEmbedder(
            model_config=cfg["model_config"],
            tokenizer=tokenizer,
            ckpt=cfg.get("ckpt"),
            batch_size=int(cfg.get("batch_size", 8)),
            device=cfg.get("device", "cpu"),
        )
    if t == "hier":
        return HierSpaFMEmbedder(
            model_config=cfg["model_config"],
            tokenizer=tokenizer,
            ckpt=cfg.get("ckpt"),
            spots_per_batch=int(cfg.get("spots_per_batch", 64)),
            device=cfg.get("device", "cpu"),
        )
    if t == "pca":
        return PCAEmbedder(n_components=int(cfg.get("n_components", 50)))
    if t == "hvg_mean":
        return HVGMeanEmbedder(top_k=int(cfg.get("top_k", 50)))
    raise ValueError(f"未知 embedder.type: {t}")


def _build_tokenizer(tok_cfg_path: str, h5ads: list[Path]) -> STTokenizer:
    tcfg = TokenizerConfig.from_yaml(tok_cfg_path)
    if tcfg.vocab_path and Path(tcfg.vocab_path).exists():
        vocab = GeneVocab.from_tsv(tcfg.vocab_path)
    else:
        symbols: list[str] = []
        seen: set[str] = set()
        for f in h5ads:
            a = read_h5ad(f)
            col = "gene_symbol" if "gene_symbol" in a.var.columns else None
            names = a.var[col].astype(str).tolist() if col else a.var_names.astype(str).tolist()
            for s in names:
                if s not in seen:
                    seen.add(s)
                    symbols.append(s)
        vocab = GeneVocab.from_symbols(symbols)
    return STTokenizer(vocab=vocab, cfg=tcfg)


def _collect_labels(files: list[Path], label_key: str) -> np.ndarray:
    out: list[int] = []
    label_to_id: dict[str, int] = {}
    for f in files:
        a = read_h5ad(f)
        if label_key not in a.obs.columns:
            raise KeyError(f"{f} 缺少 obs[{label_key!r}]")
        for v in a.obs[label_key].astype(str).tolist():
            if v not in label_to_id:
                label_to_id[v] = len(label_to_id)
            out.append(label_to_id[v])
    return np.asarray(out, dtype=np.int64)


# --------------------------------------------------------------------------- #
def _eval_imputation(emb_out: dict[str, Any], mask_ratio: float, seed: int) -> dict[str, float]:
    """对每个 cell 随机遮 ``mask_ratio`` 的 non-zero 值；用 token_repr 最后一维近似预测，
    无下游 head 时退化为 ``token_repr.mean(-1)``，仅用于 sanity（pearson 应非 0）。
    """
    rng = np.random.default_rng(seed)
    tok = emb_out.get("token_repr", [])
    vals = emb_out.get("values", [])
    if not tok or not vals:
        return {"pearson": float("nan"), "mse": float("nan"), "n": 0.0}

    preds, targets = [], []
    for h, v in zip(tok, vals, strict=True):
        v = np.asarray(v, dtype=np.float32)
        if v.size == 0:
            continue
        nz = np.flatnonzero(v != 0)
        if nz.size == 0:
            continue
        n_mask = max(1, int(round(mask_ratio * nz.size)))
        idx = rng.choice(nz, size=n_mask, replace=False)
        # 用 token_repr 的均值作为占位预测（v0 烟测）
        pred = h[idx].mean(axis=-1)
        preds.append(pred)
        targets.append(v[idx])
    if not preds:
        return {"pearson": float("nan"), "mse": float("nan"), "n": 0.0}
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    out = regression_scores(p, t)
    out["n"] = float(p.size)
    return out


# --------------------------------------------------------------------------- #
def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    files = sorted(Path(p) for p in _glob.glob(cfg.data["h5ad_glob"]))
    if not files:
        raise SystemExit(f"❌ 未匹配 h5ad：{cfg.data['h5ad_glob']}")

    # tokenizer 仅 SpaFM embedder 需要，但 baseline 也无副作用
    tokenizer = _build_tokenizer(cfg.data["tokenizer_config"], files)
    embedder = _build_embedder(cfg.embedder, tokenizer)
    emb_out = embedder.embed(files)
    X = emb_out["cell_repr"]

    results: list[dict[str, Any]] = []
    for task in cfg.tasks:
        name = task["name"]
        if name == "cell_type":
            y = _collect_labels(files, task["label_key"])
            m = linear_probe_cv(X, y, n_folds=int(task.get("cv_folds", 5)), seed=cfg.seed)
            for k, v in m.items():
                results.append({"embedder": embedder.name, "task": name, "metric": k, "value": v})
        elif name == "spatial_domain":
            y = _collect_labels(files, task["label_key"])
            m = cluster_scores(X, y, seed=cfg.seed)
            for k, v in m.items():
                results.append({"embedder": embedder.name, "task": name, "metric": k, "value": v})
        elif name == "imputation":
            m = _eval_imputation(emb_out, float(task.get("mask_ratio", 0.2)), seed=cfg.seed)
            for k, v in m.items():
                results.append({"embedder": embedder.name, "task": name, "metric": k, "value": v})
        else:
            raise ValueError(f"未知 task: {name}")

    out = {"config": {"embedder": cfg.embedder.get("type", "spafm")}, "results": results}
    json_path = cfg.output.get("json_path")
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out
