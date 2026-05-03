# Stage 2 — 表征与词表设计

> 状态：🟢 v0 实现完成
>
> 目标：把任意一份预处理后的 ST AnnData 转成模型可吃的张量序列，包含
> **gene token**、**expression value token**、**spatial position embedding**
> 三类核心信号，外加 special tokens。

---

## 1. 设计原则

1. **基因即词**：以 gene symbol 为 token 单位（参考 scGPT/Geneformer/scFoundation）。
2. **值与基因解耦**：表达量单独编码（离散 bin id 或连续标量），允许零样本迁移到不同测序深度。
3. **空间是一等公民**：每个 token 携带 (x, y[, z]) 物理坐标编码，作为附加 embedding 加在 token 上（而非简单 1D positional）。
4. **平台不可知**：词表跨 Visium / Xenium / MERFISH 共享，未在 panel 内的基因映射为 [UNK]。
5. **可流式化**：tokenizer 必须支持单细胞 / 单 spot 在线处理（用于 IterableDataset）。

---

## 2. 词表协议

### 2.1 Special tokens（固定 ID 0–7）

| ID | Token   | 用途 |
|----|---------|------|
| 0  | [PAD]   | padding |
| 1  | [CLS]   | cell / spot 级聚合表征 |
| 2  | [MASK]  | masked gene modeling 占位 |
| 3  | [UNK]   | 未登记基因 |
| 4  | [SEP]   | 多 modality 分隔 |
| 5  | [BOS]   | 序列起始（可选） |
| 6  | [EOS]   | 序列结束（可选） |
| 7  | [NICHE] | niche token，标记空间邻居聚合 |

### 2.2 Gene tokens（ID ≥ 8）

- 来源：[scripts/data/build_gene_vocab.py](../../scripts/data/build_gene_vocab.py) 输出
  的 `gene_vocab.tsv`，列 `token_id, symbol, species, source`。
- 大小写：统一 **大写 symbol**，跨物种共享同一 token id 池，物种通过单独 embedding
  注入（不为同名同源基因复制 token）。
- 容量上限：v0 设为 **64 000**。
- 未登记 gene → `[UNK]`。

### 2.3 序列化策略

每个 cell/spot 表达成长度 `L` 的序列：

```text
[CLS] g_i1 g_i2 ... g_iN  ([SEP] niche tokens ...)  → padding to L
```

挑选 `N` 个 gene 的策略（v0）：

- **top-k**：按表达量取 top-`k`（推荐 k=512 / 1024 / 2048）。
- **random-k**：随机采样 `k` 个非零基因（用于 MGM 训练增加多样性）。
- **all-nonzero**（≤ L）：FISH 类小 panel 直接全保留。

---

## 3. 表达量编码

两种可选，运行时由 config 切换：

### 3.1 离散 bin（默认）

- 对每个 cell 内的非零表达做 **log1p**，再按 cell 内 quantile 切到 `n_bins` 个 bin。
- 默认 `n_bins=51`（0 表示零，1–50 为非零分位数）。
- 输出 `value_ids ∈ [0, n_bins)`，作为独立 embedding 与 gene embedding 相加。

参考：scGPT / Geneformer 的 rank+bin 思路。

### 3.2 连续标量

- 直接输出 `log1p(count)` 浮点，模型侧通过一个小 MLP 投影到 d_model。
- 适用于 sub-cellular（Stereo-seq bin1）等极稀疏场景。

---

## 4. 空间位置编码

### 4.1 归一化

- 输入 `obsm['spatial']` 单位多样。tokenizer 配置 `coord_unit` + `coord_scale`，
  统一缩放到大致 [0, 1]（除以 `coord_scale`）。

### 4.2 编码方式

- **sin-cos 2D**：经典 NeRF 风格，对每个轴生成 `dim/4` 对 sin/cos。
- **Random Fourier Features (RFF)**：`B ~ N(0, σ²I)`，输出 `[sin(2π Bx), cos(2π Bx)]`。
- 输出维度等于 `d_model`，直接 **加** 到对应 token（包括 [CLS]，[CLS] 位置取 cell 质心）。

### 4.3 niche encoding（可选，v0 关闭）

- 借助 [src/spafm/data/graph.py](../../src/spafm/data/graph.py) 已构建的 knn 图，
  把邻居 mean spatial / mean expression 作为额外 niche token，Stage 3 详述。

---

## 5. 模块边界

```text
src/spafm/tokenization/
├── __init__.py
├── gene_vocab.py        # GeneVocab 类（含 special tokens）
├── expression.py        # bin_expression / continuous_expression
├── spatial_encoding.py  # sincos2d / rff2d
└── tokenizer.py         # STTokenizer：AnnData → dict[str, Tensor]
```

依赖关系：

- 上游：`spafm.data.vocab.load_vocab`（仅 symbol → 索引）
- 下游：`spafm.models.*`（Stage 3）通过 `STTokenizer.encode(adata)` 直接喂入 nn.Module

---

## 6. 配置示例

[configs/tokenizer/spafm-s.yaml](../../configs/tokenizer/spafm-s.yaml)：

```yaml
vocab_path: data/external/gene_vocab.tsv
max_genes: 1024
gene_select: top_k
expression:
  mode: bin
  n_bins: 51
spatial:
  mode: sincos
  dim: 128
  coord_scale: 1000.0
add_cls: true
add_niche: false
```

---

## 7. 验收清单

- [x] `GeneVocab` 含 8 special tokens + 真实基因
- [x] `bin_expression` 在 cell 内分位数稳定
- [x] `sincos2d` / `rff2d` 输出维度 / 范围正确
- [x] `STTokenizer.encode(adata)` 返回 `gene_ids / value_ids / coords / pos_emb / attention_mask`
- [x] 端到端：从 demo h5ad → tokenized batch，跑通 7 个平台
- [x] `pytest tests/tokenization/` 全绿；`make lint` 通过

---

## 8. 与后续阶段衔接

- Stage 3：`STTokenizer` 输出直接是 Hierarchical Graph Transformer 的 batch 输入。
- Stage 4：MGM 任务在 `value_ids` 上做 mask；Niche-CL 复用 `[NICHE]` token。
- Stage 5：跨模态对齐时 gene token embedding 可被替换为 LLM 文本嵌入（保持 ID 不变）。
