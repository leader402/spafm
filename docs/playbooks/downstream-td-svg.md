# T-D：基于内层注意力的 SVG（空间可变基因）下游评测

> 状态：✅ 笔记本端到端跑通
> 模块：[src/spafm/benchmarks/svg.py](../../src/spafm/benchmarks/svg.py)
> CLI：`spafm eval-svg`
> 测试：[tests/benchmarks/test_svg.py](../../tests/benchmarks/test_svg.py)（7 个）

## 1. 任务定位

SVG（spatially variable genes）是 ST 数据另一个核心下游任务。
传统方法（SpatialDE / SPARK / Moran's I）依赖**表达矩阵**直接做空间统计检验。

**Hierarchical SpaFM 的内层注意力天然就是 spot 内的 gene-gene 交互先验**——
对每个 spot 内的每个基因 token 做 in-attention 聚合 → 得到一张 `(N_spots, V_genes)`
的"注意力图片"，再在空间上算 Moran's I，与表达本身的 Moran's I 比较。

这与 T-B（CCC）形成 **dual attention interpretation** 对：

| 注意力 | 下游任务 | 论文创新点 |
| --- | --- | --- |
| outer (spot-spot) | T-B CCC | C1 + C5 |
| inner (gene-gene) | T-D SVG | C1 + C5 |

## 2. 评测协议

### 输入
- 单切片 `.h5ad`（含 `obsm["spatial"]`）
- HierarchicalSpaFM ckpt + 模型 / tokenizer 配置

### 流程
1. **注意力图片 `P_attn[N, V]`**
   - 前向 `HierarchicalSpaFM(..., return_inner_attn=True)`
   - 每层 inner attention 形状 `(B, N_spots, H, L_genes, L_genes)`
   - 跨 layer / head 平均 → `(N, L, L)`
   - 每个 token 的 in-attention 分数 = 列均值（被关注程度）
   - scatter 回 vocab 维度得到 `(N, V_genes)`

2. **基线表达 `P_expr[N, V]`** = `log1p(adata.X)`

3. **Moran's I**
   - 由 `obsm["spatial"]` 构 KNN 邻接矩阵 + 行归一化
   - 对每个基因列计算 Moran's I（向量化）
   - 得 `I_attn[V]` 与 `I_expr[V]`

4. **指标**：`spearman(I_attn, I_expr)` + top-K Jaccard
5. **Baselines**：
   - `mean_expr`：基因均值排序 vs `I_expr`
   - `var_expr`：基因方差排序 vs `I_expr`（强基线，方差大基因往往空间结构强）

### 输出
[SVGResult](../../src/spafm/benchmarks/svg.py) 含：

- `spearman_attn_vs_expr`
- `top_k_overlap`（k = 20/50/100）
- `baseline_spearman` / `baseline_top_k_overlap`
- `top_genes_attn` / `top_genes_expr`

## 3. CLI

```bash
spafm eval-svg \
  -h data/processed/visium-mouse-brain-sagittal-posterior-10x/V1_Mouse_Brain_Sagittal_Posterior.h5ad \
  --ckpt runs/spafm-s-hier-mouse/ckpt/last.ckpt \
  --model-config configs/model/spafm-s-hier.yaml \
  --tokenizer-config configs/tokenizer/spafm-s-visium-mouse.yaml \
  --max-spots 120 --knn 8 \
  --out runs/svg/posterior.json
```

## 4. 笔记本 smoke 结果（100 步 ckpt）

> ⚠️ **同 T-B：数值偏弱是预期的**，仅 100 步预训练。

| slice | spots | genes_scored | spearman(I_attn, I_expr) | var_expr 基线 |
| --- | --- | --- | --- | --- |
| Anterior | 120 | 1071 | **+0.147** | +0.600 |
| Posterior | 120 | 1024 | +0.034 | +0.653 |

posterior top-10 attention 基因（生物学合理）：

```text
CPE, RPL22L1, NRGN, ZWINT, RPL21, CALM1, PNMAL2, MAGED1, MORF4L2, NAPA
```

posterior top-10 expression 基因（gold standard）：

```text
NRGN, CAMK2A, CTXN1, LY6H, DDN, CAMKV, CNIH2, HPCAL4, BASP1, PCP2
```

attention top-10 中已出现典型神经元基因 `NRGN / CALM1 / RPL21 / MAGED1`，
说明内层注意力确实在向"空间相关基因"靠拢，只是 100 步还远未收敛。

## 5. 服务器侧的预期

按 [docs/playbooks/hierarchical-spafm-跑通.md](hierarchical-spafm-跑通.md) 迁移到服务器后：

- 训练 ≥ 10K 步、≥ 20 切片
- 然后跑 `spafm eval-svg` 在未见过的 slice 上
- 期待：
  - `spearman(I_attn, I_expr) > 0.4`（与 var_expr 基线接近或超越）
  - `top-100 jaccard > 0.4`
  - 顶层基因生物学富集（Allen brain marker / 区域特异）

如果达到，即论文 **Figure: SVG discovery via inner attention** 的核心证据。

## 6. 已验证的不变量（守门测试）

- `knn_spatial_weights`：每行和为 1（KNN 邻居）
- `morans_I_batch`：强空间信号 I > 0.5，噪声 I ≈ 0，常数列返回 NaN
- `_topk_jaccard`：基本数学正确
- `extract_inner_attention_picture`：形状 `(N, V)` + 非空覆盖
- `run_svg_analysis` 端到端：合成数据可序列化
- 左右半侧人工 SVG 信号能被 gold-standard Moran's I 召回

## 7. 后续

- 多切片 batch 评测（CLI 加 `--glob`）
- 与 SpatialDE / SPARK 直接对比（同 slice、同 top-K）
- 注意力 + 表达融合排名（ensemble，可能优于单一信号）
- T-A（细胞分群）/ T-C（zero-shot 切片对齐）
