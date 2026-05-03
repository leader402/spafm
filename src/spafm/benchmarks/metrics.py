"""评测指标——纯函数，不依赖 SpaFM 内部对象。"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import StratifiedKFold


def linear_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """对 (X, y) 做 ``n_folds`` 分层交叉验证，返回 acc / macro-F1。"""
    if len(np.unique(y)) < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    n_folds = min(n_folds, int(np.min(np.bincount(y))))
    if n_folds < 2:
        # 类别样本太少 → 退化为 train==test
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        pred = clf.predict(X)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
        }
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))}


def cluster_scores(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict[str, float]:
    """KMeans(k = 真实域数) → ARI / NMI。"""
    from sklearn.cluster import KMeans

    k = int(len(np.unique(y)))
    if k < 2 or len(y) < k:
        return {"ari": float("nan"), "nmi": float("nan")}
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    pred = km.labels_
    return {
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
    }


def regression_scores(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """整体 Pearson 相关 + MSE（pred/target 同形状，flatten 后计算）。"""
    p = np.asarray(pred).reshape(-1).astype(np.float64)
    t = np.asarray(target).reshape(-1).astype(np.float64)
    if p.size == 0:
        return {"pearson": float("nan"), "mse": float("nan")}
    if np.std(p) < 1e-12 or np.std(t) < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(p, t)[0, 1])
    mse = float(np.mean((p - t) ** 2))
    return {"pearson": pearson, "mse": mse}
