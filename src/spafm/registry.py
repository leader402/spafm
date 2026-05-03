"""SpaFM 模型注册表（Stage 8）。

为统一发布与下载占位；当前仅登记一个 spafm-s-v0（无真实下载）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCard:
    """模型卡 —— 描述一个公开发布的 SpaFM checkpoint。"""

    id: str
    size: str  # "S" | "M" | "L"
    n_params: int
    pretraining_data: str
    license: str
    download_url: str | None = None
    sha256: str | None = None
    notes: str = ""

    @property
    def status(self) -> str:
        return "available" if self.download_url else "placeholder"


MODEL_REGISTRY: dict[str, ModelCard] = {
    "spafm-s-v0": ModelCard(
        id="spafm-s-v0",
        size="S",
        n_params=2_500_000,  # 占位
        pretraining_data="Demo Visium (toy)",
        license="MIT",
        download_url=None,
        sha256=None,
        notes="开发期占位模型，仅用于测试管线，不要用于实际任务",
    ),
}


def list_models() -> list[ModelCard]:
    return list(MODEL_REGISTRY.values())


def get_model_card(model_id: str) -> ModelCard:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 ID: {model_id}（可选：{sorted(MODEL_REGISTRY)})")
    return MODEL_REGISTRY[model_id]
