# Stage 1 — 数据语料构建

> 状态：🟢 v0 + 真实样本闭环完成（acceptance 6/6 ✓ — V1_Mouse_Brain_Sagittal_Posterior 真实切片已端到端跑通）
> 责任人：core team
> 预计周期：M1–M2

本阶段是 SpaFM 项目的地基。论文反复强调：**ST FM 的成败 70% 取决于训练语料的规模、多样性与质量**。本 playbook 给出可执行清单。

---

## 1. 目标（Definition of Done）

到本阶段结束时，仓库应满足：

- [ ] **数据集登记**：[../references/数据集清单.md](../references/数据集清单.md) 收录 ≥ 30 个公开 ST 数据集，覆盖 ≥ 6 个平台、≥ 5 种组织
- [ ] **下载脚本**：每个数据集对应一个 `scripts/data/download_<dataset>.py`，可断点续传、可校验 md5
- [ ] **统一 AnnData**：所有数据落到 `data/processed/<dataset>/<sample>.h5ad`，schema 统一（详见 §4）
- [ ] **基因词表**：`data/external/gene_vocab.tsv`（人 + 鼠并集，~60k 基因）
- [ ] **空间图**：每个样本同时保存 KNN 图（k=6/8/Delaunay）到 `.obsp`
- [ ] **元数据 harmonization**：组织、平台、物种、疾病、批次 5 个核心字段在 `.obs` 与 `.uns` 中对齐
- [ ] **质量分级**：每个样本带 `tier ∈ {A, B, C}` 标签
- [ ] **CLI 跑通**：`make data-demo` 可端到端跑通至少 1 个 Visium 公开样本
- [ ] **数据卡（Data Card）**：每个数据集有 `data/processed/<dataset>/README.md`

---

## 2. 数据来源

### 2.1 测序型（spot-level）

| 平台 | 来源 | 备注 |
|---|---|---|
| Visium | 10x Genomics 官方 datasets / GEO / SODB | 主力，覆盖最广 |
| Visium HD | 10x 官方 | 2024 起，~2µm |
| Slide-seqV2 | Broad Single Cell Portal | 单细胞分辨率 |
| Stereo-seq | STOmics DB（华大） | 大场视野，亚细胞 |
| DBiT-seq | GEO | 多组学版本可选 |

### 2.2 成像型（cell-level / sub-cellular）

| 平台 | 来源 | 备注 |
|---|---|---|
| MERFISH | Vizgen MERSCOPE FFPE Atlas / Allen Brain | 高基因数 |
| Xenium | 10x 官方（人/鼠 panel） | 配套 H&E |
| CosMx | NanoString AtoMx | 1k panel |
| seqFISH+ | Cai lab | 历史数据 |
| STARmap | Wang lab | 3D ST |

### 2.3 跨平台聚合数据库

| 数据库 | URL | 价值 |
|---|---|---|
| **SODB** | <https://gene.ai.tencent.com/SpatialOmics> | 已统一 AnnData |
| **CELLxGENE** | <https://cellxgene.cziscience.com> | 含部分 ST，schema 对齐 |
| **HuBMAP** | <https://portal.hubmapconsortium.org> | 多组学，注重器官 |
| **HTAN** | <https://humantumoratlas.org> | 肿瘤 |
| **STOmics DB** | <https://db.cngb.org/stomics> | Stereo-seq 公开数据 |

### 2.4 增强 / 知识

- **scRNA**（CELLxGENE / HCA）作为 reference，用于 deconvolution 监督信号
- **基因注释**：Ensembl 111（人 GRCh38, 鼠 GRCm39）
- **基因文本知识**：NCBI Gene summary、UniProt function、Reactome、PanglaoDB、CellMarker
- **病理图像 FM**：UNI / Phikon / CONCH（用于 Stage 5）

---

## 3. 质量分级

| Tier | 标准 | 用途 |
|---|---|---|
| **A** | 专家注释完整 + ≥ 2 篇论文引用 + segmentation 可信 | 评测 / 指令微调 |
| **B** | 大规模未注释或部分注释 | 自监督预训练主力 |
| **C** | 低质量 / 合成 / 旧平台 | 仅做数据增强 |

分级写入 `.uns["spafm_tier"]`。

---

## 4. 统一 AnnData Schema

每个 sample 处理后的 `.h5ad` 必须满足：

```python
adata.X                       # 原始 counts（int32 sparse）
adata.layers["log1p_norm"]    # 归一化 + log1p
adata.layers["pearson_resid"] # 可选

adata.obs:
    - cell_id          : str
    - tier             : {"A","B","C"}
    - platform         : str  (visium / xenium / merfish / ...)
    - tissue           : str
    - species          : str  (human / mouse / ...)
    - disease          : str  (normal / cancer-... / ...)
    - donor_id         : str
    - sample_id        : str
    - n_counts         : int
    - n_genes          : int
    - pct_mt           : float
    - cell_type        : str (可空)
    - niche_label      : str (可空)

adata.var:
    - gene_id          : Ensembl ID（统一）
    - gene_symbol      : str
    - in_vocab         : bool（是否进 SpaFM 词表）

adata.obsm:
    - spatial          : (N, 2) float — 像素或 µm 坐标
    - spatial_um       : (N, 2) float — 统一单位 µm

adata.obsp:
    - knn6             : sparse (N, N) — k=6 空间近邻
    - knn8             : sparse (N, N)
    - delaunay         : sparse (N, N)

adata.uns:
    - spafm_version    : str
    - spafm_tier       : {"A","B","C"}
    - dataset_name     : str
    - source_url       : str
    - license          : str
    - preprocess       : dict — 完整 pipeline 配置回填
    - image_path       : str (可空) — 配套 H&E
```

---

## 5. 统一预处理 Pipeline

```
原始数据
   │
   ▼
[load]      平台特定 loader → AnnData
   │
   ▼
[QC]        min_counts / min_genes / max_pct_mt（按平台默认）
   │
   ▼
[segment]   仅成像型：Cellpose/Baysor 分割（可选）
   │
   ▼
[normalize] size-factor + log1p；可选 Pearson residuals
   │
   ▼
[vocab]    对齐到 SpaFM 全局基因词表（Ensembl ID）
   │
   ▼
[graph]    构建 KNN/Delaunay 空间图 → .obsp
   │
   ▼
[meta]     harmonize obs 元数据
   │
   ▼
[tier]     根据规则打质量分
   │
   ▼
[write]    保存到 data/processed/<dataset>/<sample>.h5ad
```

每一步都是 `src/spafm/data/` 下的纯函数，可独立单测。

---

## 6. 模块映射

| 模块 | 文件 | 职责 |
|---|---|---|
| 数据集 registry | `src/spafm/data/registry.py` | 数据集元信息表 + 工厂 |
| 平台 loader | `src/spafm/data/loaders/` | visium.py / xenium.py / merfish.py / ... |
| QC | `src/spafm/data/qc.py` | 过滤 spot/cell |
| 归一化 | `src/spafm/data/normalize.py` | log1p / pearson |
| 词表 | `src/spafm/data/vocab.py` | 构建与对齐 gene vocab |
| 空间图 | `src/spafm/data/graph.py` | KNN / Delaunay |
| 元数据 | `src/spafm/data/metadata.py` | harmonize obs |
| Pipeline | `src/spafm/data/pipeline.py` | 串联 step，配置驱动 |
| CLI | `src/spafm/data/cli.py` | `python -m spafm.data.cli build ...` |

---

## 7. 配置驱动

所有 pipeline 参数由 YAML 配置驱动，示例：[../../configs/data/demo-visium.yaml](../../configs/data/demo-visium.yaml)。

```bash
# 单数据集
python -m spafm.data.cli build --config configs/data/demo-visium.yaml

# 批量（合集）
python -m spafm.data.cli build-all --config configs/data/corpus-v0.yaml
```

---

## 8. 里程碑拆分

| 周 | 任务 |
|---|---|
| W1 | 完成 registry / vocab / Visium loader / QC / pipeline 骨架；跑通 demo |
| W2 | 加 Xenium / MERFISH / Slide-seq loader；批量下载脚本 |
| W3 | 元数据 harmonization；质量分级规则；空间图 |
| W4 | 完成 30 个数据集登记；构建 v0 corpus；编写每数据集 README |
| W5 | 数据可视化报告（覆盖度、组织、平台分布）；冻结 corpus-v0 |

---

## 9. 验收清单

- [x] `make data-demo` 在干净环境一键跑通
- [x] `pytest tests/data/` 全绿（19/19）
- [x] `data/processed/<dataset_id>/` 至少存在 1 个合规 `.h5ad`（demo 已落地）
- [x] 每个 loader 有最小单测（参数化覆盖 7 平台）
- [x] 数据集清单覆盖 ≥ 30 个，含 7 平台 + Visium HD + DBiT-seq
- [x] 全流程可在 [docs/guides/Stage1 复现指南.md](../guides/Stage1%20%E5%A4%8D%E7%8E%B0%E6%8C%87%E5%8D%97.md) 中复现
- [x] 至少 1 个真实样本端到端跑通并发布到 `data/processed/`（V1_Mouse_Brain_Sagittal_Posterior，3296 spots × 32285 genes，[configs/data/visium-mouse-brain-real.yaml](../../configs/data/visium-mouse-brain-real.yaml)）

---

## 10. 参考实现

- scanpy [tutorials/spatial](https://scanpy.readthedocs.io)
- squidpy [examples](https://squidpy.readthedocs.io)
- SODB Python API
- NicheCompass / Nicheformer 仓库的数据加载逻辑

---

## 附：真实数据快速上手（v0+）

```bash
# 1. 列出已配置的 Visium 直链
python scripts/data/download_visium.py --list

# 2. 下载小样本（~31MB，自动解包 spatial.tar.gz）
python scripts/data/download_visium.py \
    --dataset-id visium-mouse-brain-sagittal-posterior-10x

# 3. 跑 ingest pipeline → data/processed/.../*.h5ad
spafm data build --config configs/data/visium-mouse-brain-real.yaml

# 4. schema 校验
spafm data validate \
  data/processed/visium-mouse-brain-sagittal-posterior-10x/V1_Mouse_Brain_Sagittal_Posterior.h5ad

