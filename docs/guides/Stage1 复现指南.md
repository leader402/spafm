# Stage 1 复现指南

> 目标：在一台干净机器上 ≤30 分钟跑通 demo，≤数小时跑通真实多平台样本，得到一份可入库的 SpaFM 语料。

## 0. 前置

- Linux / macOS，Python 3.10+
- 8 GB 内存以上（demo），真实数据视样本而定
- 可选：GPU（Stage 1 不需要）

## 1. 克隆与环境

```bash
git clone <this-repo>.git
cd ST-FoundationModel

# Conda/Mamba（推荐）
mamba env create -f environment.yml -n spafm
conda activate spafm

# 或 venv + pip
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## 2. 跑 demo（不下载任何数据）

```bash
spafm data list-datasets                       # 浏览 30+ 已登记数据集
spafm data build --config configs/data/demo-visium.yaml
spafm data validate data/processed/visium-mouse-brain-sagittal-10x/demo_synth_001.h5ad
```

预期输出：6 步日志全绿，`data/processed/.../demo_synth_001.h5ad` 体积约 100 KB，
`schema 通过` 行出现即视为成功。

## 3. 跑多平台 demo 语料

```bash
spafm data build-all --corpus configs/data/corpus-v0.yaml
```

会按 corpus 中列出的 7 个平台各跑一次合成 pipeline，全部成功后会在
`data/processed/<dataset_id>/demo_<platform>.h5ad` 各落地一个文件。

## 4. 跑真实样本（以 Visium 为例）

```bash
# 1) 下载（10x 官方源；--mirror 指定镜像）
python scripts/data/download_visium.py \
    --dataset-id visium-mouse-brain-sagittal-10x \
    --output-dir data/raw/visium-mouse-brain-sagittal-10x

# 2) 写一份样本配置（基于 demo-visium.yaml 修改）
cat > configs/data/visium-mouse-brain-real.yaml <<'EOF'
dataset_id: visium-mouse-brain-sagittal-10x
sample_id: V1_Mouse_Brain_Sagittal_Anterior
input_path: data/raw/visium-mouse-brain-sagittal-10x
output_dir: data/processed
donor_id: 10x_demo_donor
disease: normal
qc: {min_genes: 200, max_pct_mt: 25.0}
normalize: log1p
graph_ks: [6, 8]
EOF

# 3) 构建
spafm data build --config configs/data/visium-mouse-brain-real.yaml
```

## 5. 构建全局 gene vocab

在跑完一定数量样本后：

```bash
python scripts/data/build_gene_vocab.py build \
    --h5ad-glob "data/processed/**/*.h5ad" \
    --output data/external/gene_vocab.tsv
```

如有 Ensembl GTF，可 `--human-gtf path.gtf.gz --mouse-gtf path.gtf.gz` 一并并入。

后续在 pipeline 配置中加 `vocab_path: data/external/gene_vocab.tsv` 即可使用统一词表对齐。

## 6. 质量门禁

```bash
make lint       # ruff + black + markdownlint
make test       # pytest tests/
```

## 7. 已知问题

| 现象 | 解决 |
|---|---|
| `make data-demo` 提示 anndata 版本警告 | 使用 anndata≥0.10 即可，警告无害 |
| Stereo-seq GEM 太大内存爆 | 调大 `bin_size`（如 100），或预先 `gem2gef` |
| Xenium `cell_feature_matrix.h5` 缺失 | 旧版本 bundle 用 `cell_feature_matrix/` 目录，TODO 后续支持 mtx 格式 |

## 8. Stage 1 验收清单

- [x] ≥30 数据集登记
- [x] 7 个平台真实 loader 实现
- [x] 单样本 pipeline + 批量 pipeline 跑通 demo
- [x] 全局 gene vocab 构建脚本
- [x] `make lint` / `make test` 全绿
- [ ] 至少 1 个真实样本端到端跑通并发布到 `data/processed/`（依赖人工下载）
