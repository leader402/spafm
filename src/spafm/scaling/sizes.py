"""SpaFM 三档预设规模（与 yaml 对齐的内存版）。"""

from __future__ import annotations

from spafm.models import ModelConfig

SIZE_CONFIGS: dict[str, ModelConfig] = {
    "S": ModelConfig(vocab_size=64000, d_model=256, d_pos=128, n_layers=6, n_heads=4, d_ffn=1024),
    "M": ModelConfig(vocab_size=64000, d_model=512, d_pos=256, n_layers=12, n_heads=8, d_ffn=2048),
    "L": ModelConfig(
        vocab_size=64000, d_model=1024, d_pos=512, n_layers=24, n_heads=16, d_ffn=4096
    ),
}


def get_size_config(size: str) -> ModelConfig:
    s = size.upper()
    if s not in SIZE_CONFIGS:
        raise KeyError(f"未知规模 {size}（可选 {sorted(SIZE_CONFIGS)}）")
    return SIZE_CONFIGS[s]
