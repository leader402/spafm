# SpaFM — 空间转录组学基础模型（Spatial Transcriptomics Foundation Model）

> 一份从想法到模型的端到端工程实践。

本仓库的目标是构建一个面向**空间转录组学（ST）**数据的基础模型 **SpaFM**，覆盖：

- 多平台 ST 数据（Visium / Visium HD / Slide-seqV2 / Stereo-seq / MERFISH / Xenium / CosMx / STARmap …）的统一语料构建
- 序列驱动（seq-based）+ 知识驱动（knowledge-based）混合预训练
- 空间感知 Transformer 架构（Graph-aware + 多模态）
- 全套下游任务评测框架（low / mid / high-level benchmark）

立项依据：Liu et al., *A perspective on developing foundation models for analyzing spatial transcriptomic data*, **Quantitative Biology**, 2025。要点解读见 [docs/concepts/论文要点解读.md](docs/concepts/论文要点解读.md)。

---

## 🚀 快速开始

```bash
# 1. 拉取代码
git clone <this-repo> && cd ST-FoundationModel

# 2. 创建 Python 环境（推荐 conda / mamba）
mamba env create -f environment.yml
conda activate spafm

# 3. 查看当前阶段任务
cat docs/playbooks/stage1-数据语料构建.md

# 4. 运行最小数据 pipeline（示例 Visium 数据）
make data-demo

# 5. 质量检查
make lint
make test
```

完整环境配置见 [docs/getting-started/环境配置.md](docs/getting-started/环境配置.md)。

---

## 🗺️ 项目地图

```
.
├── README.md                       # 本文件，面向人类开发者
├── CONTRIBUTING.md                 # 贡献指南
├── LICENSE                         # MIT 许可证
├── Makefile                        # 自动化命令入口
├── environment.yml                 # Conda 环境定义
├── pyproject.toml                  # Python 项目元数据 + 依赖
│
├── docs/                           # 知识库（中文）
│   ├── getting-started/            # 入门：环境、数据、最小示例
│   ├── concepts/                   # 核心概念：FM 范式、论文要点、术语
│   ├── guides/                     # 操作指南：开发流程、调试、复现
│   ├── playbooks/                  # 阶段执行手册（stage1 ~ stage10）
│   └── references/                 # 技术参考：基因词表、平台对照表、引文
│
│
├── src/spafm/                      # 模型主代码（Python 包）
│   ├── data/                       # 数据加载、预处理、词表
│   ├── models/                     # 架构（编码器、注意力、损失）
│   ├── training/                   # 训练循环、调度、回调
│   ├── benchmarks/                 # 下游任务评测
│   └── utils/                      # 通用工具
│
├── configs/                        # YAML 配置（数据 / 模型 / 训练）
│   ├── data/
│   ├── model/
│   └── training/
│
├── data/                           # 数据目录（默认 gitignore，仅留说明）
│   ├── raw/                        # 原始下载
│   ├── processed/                  # AnnData / Zarr
│   └── external/                   # 外部 reference（基因注释等）
│
├── scripts/                        # 一次性脚本与自动化
│   ├── data/                       # 数据下载、清洗、构建
│   └── backups/                    # 备份脚本
│
├── notebooks/                      # 探索性分析（可选 Jupyter）
├── tests/                          # 单元测试 / 集成测试
├── benchmarks/                     # 评测结果与对比基线
├── paper/                          # 立项原文 PDF
└── .github/workflows/              # CI（lint + test）
```

---

## 🧭 当前阶段

| 阶段 | 名称 | 状态 |
|---|---|---|
| **Stage 1** | **数据语料构建** | 🟢 v0 完成（7 平台 loader + 30 数据集 + corpus build-all）|
| **Stage 2** | **表征与词表设计** | 🟢 v0 完成（`spafm.tokenization.STTokenizer` + special tokens + bin/sincos）|
| **Stage 3** | **模型架构（SpaFM-S 原型）** | 🟢 v0 完成（`SpaFMModel` + spatial-bias attention + GEGLU FFN + tied MGM head）|
| **Stage 4** | **多任务自监督预训练** | 🟢 v0 完成（Lightning + MGM + 对比学习 + smoke test）|
| **Stage 5** | **知识增强与跨模态对齐** | 🟢 v0 完成（基因先验嵌入对齐 + cosine alignment loss）|
| Stage 6 | 下游适配（LoRA / Linear Probe / Fine-tune） | 🟢 v0 完成（三种策略 × 三种 head：cell_type / spatial_domain / imputation）|
| Stage 7 | 评测基准 | 🟢 v0 完成（metrics + SpaFM/PCA/HVG embedder + run_benchmark CLI）|
| Stage 8 | 工程化与开源发布 | 🟢 v0 完成（统一 `spafm` CLI + ModelRegistry + ModelCard + CI matrix）|
| Stage 9 | SpaFM-L 扩展与 Scaling Law | 🟢 v0 完成（S/M/L 三档 + param/FLOPs 估计 + scaling-law 拟合）|

每个阶段对应 [docs/playbooks/](docs/playbooks/) 中的一份执行手册。

---

## 📐 设计原则

1. **规划驱动**：每个阶段先有 playbook，再有代码。
2. **模块化**：`data / models / training / benchmarks` 严格分层，互不耦合。
3. **可复现**：固定随机种子；所有实验配置存 [configs/](configs/)；所有数据来源记录在 [docs/references/数据集清单.md](docs/references/数据集清单.md)。
4. **质量门禁**：`make lint` + `make test` 必须通过才能合入。
5. **不重复造轮子**：优先复用 `scanpy / squidpy / anndata / scvi-tools / lightning` 生态。

---

## 🤝 贡献

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 📚 引用

立项依据：

> Liu T, Hao M, Liu X, Zhao H. A perspective on developing foundation models for analyzing spatial transcriptomic data. *Quantitative Biology*. 2025;e70010. <https://doi.org/10.1002/qub2.70010>

## 📜 License

MIT License — 详见 [LICENSE](LICENSE)。
