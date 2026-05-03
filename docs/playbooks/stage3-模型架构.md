# Stage 3 — 模型架构（SpaFM-S 原型）

> 状态：🟢 v0 实现完成
>
> 目标：在 [Stage 2](stage2-表征与词表设计.md) tokenizer 之上，搭起一个最小可训练
> 的 **SpaFM-S** 模型骨架（~10–80M 参数），能直接吃 `STTokenizer.encode()` 输出的
> batch 字典，输出 token 表征 + cell 表征 + 可选预训练 logits。

---

## 1. 总体结构

```text
                 ┌──────────── inputs ───────────┐
gene_ids   ─────►│ GeneEmbedding (V, d)          │
value_ids  ─────►│ ValueEmbedding (Nb, d)        │
pos_emb    ─────►│ Linear(d_pos → d)             │
                 └──────────────┬────────────────┘
                                ▼
              token_emb = sum(.) + LayerNorm
                                ▼
             ┌──────────────────┴──────────────────┐
             │  N × SpatialTransformerBlock        │
             │  ├── MultiHeadSelfAttention         │
             │  │     + spatial-distance bias B    │
             │  ├── Add & Norm                     │
             │  └── FFN (GEGLU/MLP) + Add & Norm   │
             └──────────────────┬──────────────────┘
                                ▼
              token_repr  ──►  GeneHead (MLM)            (Stage 4)
              CLS_repr    ──►  CellHead / ContrastHead
```

---

## 2. 关键决策

| 项目 | 选择 | 备注 |
|---|---|---|
| 维度 d_model | 256 (S) / 512 (B) / 1024 (L) | 预设 |
| 层数 | 6 / 12 / 24 | 预设 |
| 头数 | 4 / 8 / 16 | d_head = 64 |
| FFN | GEGLU | 表现优于 ReLU/GELU MLP |
| LayerNorm | Pre-LN | 训练稳定 |
| Attention | 标准 MHSA + 可选 spatial bias | 有 `flash_attn` 时自动启用 |
| Spatial bias | $B_{ij} = -\alpha \cdot \\|x_i - x_j\\|^2 / \sigma^2$ | 单 head 共享或可学习每头 $\alpha$ |
| pos_emb | 由 tokenizer 给出 (sincos / RFF) | 经一层 Linear 投到 d |
| value 编码 | bin → Embedding；continuous → MLP(1→d) | 两条路径同结构 |
| 共享词表 embedding | 与 MGM head 是否绑权 | v0 默认绑权（Tie） |

> **图意识**：v0 的 spatial bias 已经利用了 token 的物理坐标差，相当于"距离感知注意力"；
> Stage 4 会引入更显式的 KNN 图 attention mask；Stage 5 把 H&E patch 接入。

---

## 3. 输入契约

输入为 [STTokenizer.encode](../../src/spafm/tokenization/tokenizer.py) 的输出字典，被一个轻量的
`batch_to_tensors` helper 转成 torch.Tensor。模型 `forward` 期望的字段：

| key | shape | dtype | 必需 |
|---|---|---|---|
| `gene_ids` | (B, L) | int64 | ✓ |
| `value_ids` | (B, L) | int64 | bin 模式 |
| `value_floats` | (B, L) | float32 | continuous 模式 |
| `coords` | (B, L, 2) | float32 | ✓（用于 spatial bias） |
| `pos_emb` | (B, L, d_pos) | float32 | ✓ |
| `attention_mask` | (B, L) | bool | ✓ |

返回字典：

| key | shape | 说明 |
|---|---|---|
| `token_repr` | (B, L, d) | 每个 token 的最后层表示 |
| `cell_repr` | (B, d) | CLS 处的 cell/spot embedding |
| `gene_logits` | (B, L, V) | （可选）MGM head 输出，仅 `return_gene_logits=True` |

---

## 4. 模块边界

```text
src/spafm/models/
├── __init__.py
├── embedding.py          # GeneEmbedding + ValueEmbedding + Composer
├── attention.py          # MultiHeadSelfAttention + spatial bias
├── transformer.py        # SpatialTransformerBlock + 堆叠
├── heads.py              # MGMHead / ContrastiveHead
├── spafm.py              # SpaFMModel + ModelConfig
└── utils.py              # batch_to_tensors, count_parameters
```

---

## 5. 配置

[configs/model/spafm-s.yaml](../../configs/model/spafm-s.yaml)：

```yaml
vocab_size: 64000
n_value_bins: 51
expression_mode: bin            # bin | continuous
d_model: 256
d_pos: 128
n_layers: 6
n_heads: 4
d_ffn: 1024
dropout: 0.1
spatial_bias: {enabled: true, sigma: 200.0}
tie_gene_embedding: true
```

---

## 6. 验收清单

- [x] `SpaFMModel.from_config(yaml).forward(batch)` 端到端跑通 demo h5ad
- [x] 模型参数量 ≈ 10–15M（spafm-s）
- [x] `gene_logits.shape == (B, L, vocab_size)`
- [x] CPU 上 batch=4, L=64 单步 < 1s
- [x] `pytest tests/models/` 全绿；`make lint` 通过

---

## 7. 与后续阶段衔接

- Stage 4：在 `SpaFMModel` 之上实现 MGM / NCL / GG-Recovery loss + Lightning Module。
- Stage 5：替换 `GeneEmbedding` 权重为 LLM 文本 embedding，做 zero-shot eval。
- Stage 6：冻结 backbone + 加 LoRA / Linear Probe head。
