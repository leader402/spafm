# SpaFM 总体方案

> 端到端蓝图：从数据 → 模型 → 训练 → 评测 → 部署。详细阶段执行手册见 [../playbooks/](../playbooks/)。

---

## 总览

```
┌──────────────────────────────────────────────────────────────┐
│  SpaFM: Hybrid Seq + Knowledge Foundation Model for ST       │
├──────────────────────────────────────────────────────────────┤
│ 数据层 ：多平台 ST 语料 + scRNA 增强 + H&E 图像 + 文本知识    │
│ 表征层 ：Gene token + Cell/Spot token + Spatial token + Image │
│ 架构层 ：Graph-aware Hierarchical Transformer (Perceiver-IO)  │
│ 训练层 ：MGM + Niche CL + Spatio-Temporal Traj + GG-Recovery  │
│ 对齐层 ：Image↔Expression / Text↔Gene CLIP-style              │
│ 下游层 ：Zero-shot embedding / LoRA / Linear Probe    │
│ 评测层 ：Low / Mid / High-level + 强制 baseline 比对          │
└──────────────────────────────────────────────────────────────┘
```

---

## 阶段路线图

| 阶段 | 名称 | 关键交付 | Playbook |
|---|---|---|---|
| 1 | 数据语料构建 | ≥1 亿 cell/spot；统一预处理 pipeline；基因词表 | [stage1](../playbooks/stage1-数据语料构建.md) |
| 2 | 表征与词表设计 | gene/spatial/cell token 编码方案 | [stage2](../playbooks/stage2-表征与词表设计.md) |
| 3 | 模型架构（SpaFM-S） | Hierarchical Graph Transformer 原型 | [stage3](../playbooks/stage3-模型架构.md) |
| 4 | 多任务自监督预训练 | MGM + NCL + ST-Traj + GG-Recovery | [stage4](../playbooks/stage4-预训练目标.md) |
| 5 | 知识增强与跨模态对齐 | LLM gene embedding；图像 CLIP | [stage5](../playbooks/stage5-知识增强.md) |
| 6 | 下游适配 | LoRA / Linear Probe / In-context | [stage6](../playbooks/stage6-下游适配.md) |
| 7 | SpaFM-Bench | 多层评测 + 强制 baseline 比对 | [stage7](../playbooks/stage7-评测框架.md) |
| 8 | 工程化与开源 | HuggingFace 权重 + Gradio Demo | [stage8](../playbooks/stage8-工程化.md) |
| 9 | SpaFM-L 扩展 | Scaling law 实验 | [stage9](../playbooks/stage9-scaling.md) |

---

## 模型规模规划

| 名称 | 参数量 | 用途 | 训练资源（估算） |
|---|---|---|---|
| **SpaFM-S** | ~80M | 消费级 GPU 可微调 | 4× A100-40G × 1 周 |
| **SpaFM-B** | ~350M | 主力模型 | 8× A100-80G × 2 周 |
| **SpaFM-L** | ~1.3B | Scaling law 探索 | 32× H100 × 4 周 |

---

## 风险对照表

| 风险 | 来源 | 对策 |
|---|---|---|
| 性能不及简单方法 | 论文 [16] | 每个里程碑必须过 baseline |
| 预训练目标无效 | 论文 §4 | 消融实验量化每个 loss |
| 数据偏倚 | 公开数据集主要为人/小鼠肿瘤 | 主动收集发育、神经、植物 |
| 算力受限 | 实验室预算 | 优先 SpaFM-S；teacher 蒸馏 |
| FM 称号不实 | 论文警示 | 锁定 2 个旗舰高级任务（non-contiguous 域、空间扰动）|
