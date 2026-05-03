"""模型层小工具。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


def batch_to_tensors(
    batch: dict[str, np.ndarray], device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """把 ``STTokenizer.encode`` 返回的 numpy 字典转为同名 torch.Tensor。

    - bool / int → 原 dtype
    - float → float32
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if not isinstance(v, np.ndarray):
            continue
        if v.dtype == bool:
            t = torch.from_numpy(v.copy())
        elif np.issubdtype(v.dtype, np.integer):
            t = torch.from_numpy(v.astype(np.int64, copy=False))
        else:
            t = torch.from_numpy(v.astype(np.float32, copy=False))
        out[k] = t.to(device)
    return out


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """统计模块参数量。"""
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def merge_config(default: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """浅合并配置字典，``override`` 优先。"""
    out = dict(default)
    if override:
        out.update(override)
    return out
