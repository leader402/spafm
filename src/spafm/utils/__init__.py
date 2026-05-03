"""通用工具函数。"""

from __future__ import annotations

import logging
import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """固定 numpy / random / torch（如果安装）的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "spafm", level: int = logging.INFO) -> logging.Logger:
    """构建带 rich 格式的 logger。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, markup=True)
    except ImportError:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
