# Stage 5 — 知识增强与跨模态对齐

> 状态：🟢 v0 完成（基因先验嵌入对齐 + 集成进 PretrainModule）
>
> 上游：[stage4-预训练目标.md](stage4-预训练目标.md)　下游：[stage6-下游适配.md](stage6-下游适配.md)

---

## 1. 目标

Stage 4 的自监督只用了"语料里基因共现"这一种信号。Stage 5 引入**外部先验知识**进一步约束模型表征：

| 知识源 | 形态 | v0 实现 | 后续扩展 |
|---|---|---|---|
| **基因 LLM 嵌入**（GenePT / scGPT prior / Gene Ontology embedding 等）| `gene_symbol → R^{d_prior}` 字典 | ✅ `GenePriorBank` 加载 + `PriorAligner` cosine 对齐 | 直接接 GenePT-3 768d，或 ESM-2 蛋白嵌入 |
| **通路 / 模块** | `pathway_id → [gene_symbol, ...]` | ⚪ 占位（`pathway_contrastive` 接口已留） | KEGG / Reactome / Hallmark |
| **H&E 图像 patch** | `cell_id → R^{d_img}`（DINO/UNI 提取） | ⚪ 留接口；笔记本无图，跳过 | Stage 6+ 在服务器侧加 |
| **细胞类型本体（CL）** | `cell_id → cl_term` | ⚪ 留作下游评测使用 | |

**核心设计**：把所有知识都抽象成"`gene_symbol → 向量`"或"`gene_symbol → 类别`"两种 schema，模型主干完全不变；只在训练时多加一项 loss。

---

## 2. 先验对齐 loss

设模型 GeneEmbedding 矩阵 $E \in \mathbb{R}^{V \times d}$，先验矩阵 $P \in \mathbb{R}^{V \times d_p}$（缺失的 gene 用 mask=False 跳过）。

引入投影 $W \in \mathbb{R}^{d \times d_p}$，把 $E W$ 投到先验空间，再做 cosine 损失：

$$
\mathcal{L}_\text{align}
= \frac{1}{|\mathcal{M}|}\sum_{g \in \mathcal{M}}
\bigl(1 - \cos(E_g W,\ P_g)\bigr)
$$

其中 $\mathcal{M} = \{g\ |\ \text{prior\_mask}_g = \text{True}\}$。

总损失（叠加在 Stage 4 上）：

$$
\mathcal{L} = \lambda_\text{mgm}\mathcal{L}_\text{MGM}
+ \lambda_\text{ccl}\mathcal{L}_\text{CCL}
+ \lambda_\text{align}\mathcal{L}_\text{align}
$$

v0 默认 $\lambda_\text{align}=0.1$。

**为什么用投影而不是直接 $d=d_p$**：先验维度（GenePT=1536 / 768）通常远大于模型 d_model；投影既保持模型容量，又允许冻结 / 不冻结先验空间。

---

## 3. 模块边界

| 模块 | 职责 |
|---|---|
| [knowledge/gene_priors.py](../../src/spafm/knowledge/gene_priors.py) | `GenePriorBank.from_tsv` / `from_npz` 加载先验；`align_to_vocab(vocab)` 返回 `(V, d_p)` 张量 + `(V,)` bool mask |
| [knowledge/aligner.py](../../src/spafm/knowledge/aligner.py) | `PriorAligner(d_model, d_prior)` 含投影；`alignment_loss(gene_embedding, prior_matrix, mask)` |
| [knowledge/synth.py](../../src/spafm/knowledge/synth.py) | 生成可复现的合成先验，便于单测与 demo |
| [training/lit_module.py](../../src/spafm/training/lit_module.py) | 在 `SpaFMPretrainModule` 中可选挂载 `PriorAligner`，加 alignment_loss |

---

## 4. 配置示例

新增字段（[configs/training/spafm-s-pretrain.yaml](../../configs/training/spafm-s-pretrain.yaml)）：

```yaml
knowledge:
  enabled: false                 # 默认关，避免破坏 Stage 4 行为
  prior_path: data/external/gene_prior_demo.npz
  prior_format: npz              # npz | tsv
  alignment_weight: 0.1
  freeze_prior: true             # 是否在训练中冻结 P
```

`gene_prior_demo.npz` 由 [scripts/data/build_gene_prior_demo.py](../../scripts/data/build_gene_prior_demo.py) 生成，存 `{symbols: (N,) U-string, vectors: (N, d_p) float32}`。

---

## 5. 验收清单

- [x] `GenePriorBank.from_npz` / `from_tsv` 加载与 vocab 对齐
- [x] `PriorAligner` cosine 对齐 loss + 投影矩阵
- [x] `SpaFMPretrainModule` 支持 `knowledge.enabled=true` 时附加 alignment loss
- [x] `scripts/data/build_gene_prior_demo.py` 可生成可复现 demo 先验
- [x] 单元测试覆盖：加载/对齐/loss/Lightning fit
- [ ] 接入真实 GenePT prior（待服务器侧执行，方法见 §6）

---

## 6. 接入真实 GenePT prior（服务器侧）

```bash
# 1. 下载 GenePT-3 嵌入（约 1.5 GB，需 HF token）
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('NYUAD-CAI/GenePT', local_dir='data/external/genept')"

# 2. 转换成本仓库 npz schema
python scripts/data/convert_genept_to_npz.py \
  --input data/external/genept/gene_embeddings.json \
  --output data/external/gene_prior_genept.npz

# 3. 训练
python -m scripts.train_pretrain -c configs/training/spafm-s-pretrain.yaml \
  -o knowledge.enabled=true \
  -o knowledge.prior_path=data/external/gene_prior_genept.npz \
  -o knowledge.alignment_weight=0.1
```

`convert_genept_to_npz.py` 在 v0 暂未实现，留作 TODO。
