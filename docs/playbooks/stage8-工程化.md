# Stage 8 · 工程化与开源发布（v0）

## 目标

把前 7 个 stage 的能力打包成"**克隆即可用、装上即可跑**"的开源项目：

1. **统一 CLI**：`spafm` 顶层命令聚合 `data / pretrain / finetune / eval / model` 五大子命令
2. **版本与元数据**：`spafm.__version__`、`spafm version` 命令
3. **模型注册表**：`spafm.registry.MODEL_REGISTRY` + `spafm model list/info`
4. **模型卡模板**：`docs/model-cards/spafm-s.md`
5. **CI**：`.github/workflows/ci.yml` 跑 lint + test

> **不做** 的事：真正的 PyPI/Docker 发布（留 v1）。

---

## 模块布局

```
src/spafm/
├── __init__.py             # __version__
├── cli.py                  # 顶层 spafm 命令（聚合所有子命令）
├── registry.py             # ModelCard + MODEL_REGISTRY
├── data/cli.py             # 已有
├── training/cli.py         # 新增：spafm pretrain
├── adaptation/cli.py       # 新增：spafm finetune
└── benchmarks/cli.py       # 新增：spafm eval
```

---

## 顶层命令

```
$ spafm --help
spafm
├── version
├── data
│   ├── build / build-all / list-datasets
├── pretrain   -c <yaml>  [-o k=v]
├── finetune   -c <yaml>  [-o k=v]
├── eval       -c <yaml>  [-o k=v]
└── model
    ├── list
    └── info ID
```

老的 `scripts/*.py` 仍可独立运行（`python -m scripts.train_pretrain ...`），CLI 入口
内部直接复用同一份 `main` 函数。

---

## 模型注册表

```python
@dataclass(frozen=True)
class ModelCard:
    id: str
    size: str                  # S | M | L
    n_params: int
    pretraining_data: str
    license: str
    download_url: str | None
    sha256: str | None
    notes: str = ""
```

`MODEL_REGISTRY` 当前只登记 `spafm-s-v0`（占位，无下载地址）。

---

## 验收清单

- [x] `spafm --help` 显示 5 个子命令组
- [x] `spafm version` 打印 `__version__`
- [x] `spafm model list` / `spafm model info spafm-s-v0`
- [x] `spafm eval -c configs/benchmark/spafm-s-eval.yaml -o embedder.type=pca`
- [x] `make test` 全绿（含 `tests/cli/`）

---

## 后续（v1+）

- PyPI 发布脚本 + 自动版本号
- Docker 镜像（CPU + CUDA）
- Hugging Face Hub 上传 ckpt
- 文档站点（mkdocs-material）
