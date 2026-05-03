# T-B：基于外层注意力的 CCC（细胞-细胞通讯）下游评测

> 状态：✅ 笔记本端到端跑通
> 模块：[src/spafm/benchmarks/ccc.py](../../src/spafm/benchmarks/ccc.py)
> CLI：`spafm eval-ccc`
> 测试：[tests/benchmarks/test_ccc.py](../../tests/benchmarks/test_ccc.py)（7 个）

## 1. 任务定位

CCC（cell-cell communication）是 ST 数据最具代表性的下游任务之一。
传统方法（CellChat / NicheNet）依赖人工 L-R 数据库 + 表达共发生计算，
**Hierarchical SpaFM 的外层注意力天然就是 spot-spot 通讯先验**。

这是论文叙事中 **C1（spatial-aware attention）** + **C5（attention 可解释为生物先验）**
两个创新点的关键证据。

## 2. 评测协议

### 输入
- 一个切片 `.h5ad`（含 `obsm["spatial"]` 与 `obs[label_key]`）
- HierarchicalSpaFM ckpt + 模型 / tokenizer 配置
- L-R 对列表（默认内置 8 对小鼠脑常用 L-R）

### 流程
1. **SpaFM 通讯矩阵 `M_attn[K,K]`**
   - 前向 `HierarchicalSpaFM(..., return_attn=True)`，得 outer attentions `list[(1,H,N,N)]`
   - 跨 layer / head 平均 → spot-spot 矩阵 `A[N,N]`
   - 按 cell-type 标签聚合到 `(K, K)`（cell-type 间通讯强度）

2. **L-R 共表达矩阵 `M_lr[K,K]`**
   - 对每对 `(L, R)`：`spot_mat[s,t] = log1p(L,s) * log1p(R,t)` → 聚合到 `(K,K)`
   - 所有 LR 对取均值

3. **指标**：`M_attn` 与 `M_lr` 的 Spearman / Pearson 相关
4. **Baselines**：
   - `pca_cosine`：PCA(50) embedding 的余弦相似度
   - `rbf_spatial`：纯 spot 距离 RBF kernel（不看表达）

### 输出
[CCCResult](../../src/spafm/benchmarks/ccc.py#L82) 含：

- `overall_corr_spearman / pearson`
- `per_lr_spearman`（每对 LR 单独）
- `baseline_corrs`
- `M_attn` / `M_lr` 矩阵（可序列化）

## 3. CLI

```bash
spafm eval-ccc \
  -h data/processed/visium-mouse-brain-sagittal-posterior-10x/V1_Mouse_Brain_Sagittal_Posterior.h5ad \
  --ckpt runs/spafm-s-hier-mouse/ckpt/last.ckpt \
  --model-config configs/model/spafm-s-hier.yaml \
  --tokenizer-config configs/tokenizer/spafm-s-visium-mouse.yaml \
  --label-key cell_type \
  --max-spots 400 \
  --device cpu \
  --out runs/ccc/posterior.json
```

## 4. 笔记本 smoke 结果（100 步 ckpt）

> ⚠️ **结果数值偏弱是预期的**：仅 100 步预训练、2 切片、cell_type 是无监督 `cluster_*`。
> 跑通流程才是当前阶段目标，绝对数值需服务器长程训练后再评估。

posterior 切片，max_spots=200：

```text
n_spots=199  n_celltypes=13  L-R 对：8/8 全部命中
overall  spearman=-0.125  pearson=+0.100
baselines:  pca_cosine=-0.095   rbf_spatial=-0.065
```

模型与 baseline 都接近 0（相当于随机），说明 **100 步 ckpt 还没真正学到通讯先验**——
但 pipeline、CLI、json 输出、L-R 命中、矩阵聚合全部正常。

## 5. 服务器侧的预期

按 [docs/playbooks/hierarchical-spafm-跑通.md](hierarchical-spafm-跑通.md) 迁移到服务器后：

- 训练 ≥ 10K 步、≥ 20 切片
- 然后跑 `spafm eval-ccc` 在多个未见过的 slice 上
- 期待：
  - `overall_corr_spearman > 0.2`（比 PCA 基线高 ≥ 50%）
  - `per_lr_spearman` 出现明显的 `Tgfb1-Tgfbr1`、`Cxcl12-Cxcr4` 等强信号 LR 对

如果达到这个数字，即可作为论文 **Figure: CCC discovery without supervision** 的核心证据。

## 6. 已验证的不变量（守门测试）

- `_aggregate_to_celltype`：(N,N)→(K,K) 聚合数学正确
- `extract_outer_attention`：每行 softmax 和约为 1
- `lr_coexpression_matrix`：缺失基因返回 None，存在则 (N,N)
- `baselines`：对角线归一
- `run_ccc_analysis` 端到端：合成数据 60 cells × 26 genes 一遍跑通，结果可序列化

## 7. 后续

- T-D：基于 inner attention 的 SVG（空间可变基因）发现
- 多切片 batch 评测（CLI 加 `--glob` 支持）
- L-R 数据库扩展（CellChatDB / OmniPath 完整版）
