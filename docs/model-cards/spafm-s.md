# Model Card · spafm-s-v0

> **Status**：开发期占位，**不要**用于实际任务。

## 概览

| 字段 | 值 |
|---|---|
| ID | `spafm-s-v0` |
| 规模 | S（小型） |
| 参数量 | ~2.5M（占位） |
| 架构 | SpaFMModel：spatial-bias attention + GEGLU FFN + tied MGM head |
| 预训练数据 | demo Visium toy（仅用于跑通 pipeline） |
| 预训练目标 | MGM (Masked Gene Modeling) + 对比学习 + 知识对齐 |
| License | MIT |

## 训练配置

详见 [configs/training/spafm-s-pretrain.yaml](../../configs/training/spafm-s-pretrain.yaml)。

## 适配方式

支持三种下游适配（见 [docs/playbooks/stage6-下游适配.md](../playbooks/stage6-下游适配.md)）：

- `linear_probe`
- `lora`（默认）
- `full`

## 下游评测

详见 [docs/playbooks/stage7-评测基准.md](../playbooks/stage7-评测基准.md)。

## 已知局限

- 当前权重未经过真实数据预训练
- 词表为 demo 自动构建，**不**包含完整人/鼠基因 panel
- 空间 bias 仅支持 2D 单切片

## 引用

```bibtex
@misc{spafm2026,
  title  = {SpaFM: A Spatial Transcriptomics Foundation Model},
  author = {SpaFM Contributors},
  year   = {2026},
  url    = {https://github.com/<org>/ST-FoundationModel}
}
```
