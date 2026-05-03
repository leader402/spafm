# Stage 7 · 评测基准（v0）

## 目标

为 SpaFM 提供**统一、可复现的下游评测**：从 `.h5ad`（带标签）→
模型 / baseline 嵌入 → 任务指标 → 结果表（json + console）。

不依赖外部数据下载，最小数据用 [data/processed/](../../data/processed/)
里既有的 demo h5ad；服务器上换大数据只改 yaml 即可。

---

## 三类任务 × 指标

| 任务 | 输入 | 模型输出 | 指标 |
|---|---|---|---|
| `cell_type` 细胞类型注释 | `obs[label_key]` | `cell_repr` → 线性分类器（5-fold CV） | accuracy, macro-F1 |
| `spatial_domain` 空间域识别 | `obs[label_key]`（域标签） | `cell_repr` → KMeans(k=域数) | ARI, NMI |
| `imputation` 表达插补 | 随机遮 20% non-zero 基因 | `token_repr` → 预测原值 | Pearson, MSE |

---

## 评估对象（embedder）

- **SpaFM** ：加载 ckpt，前向取 `cell_repr` / `token_repr`
- **Baseline-PCA** ：log1p → 50 维 PCA
- **Baseline-HVG-mean** ：top-k HVG 求平均（最弱基线）

baseline 不依赖 ckpt，作为 sanity floor。

---

## 模块布局

```
src/spafm/benchmarks/
├── __init__.py
├── metrics.py       # 纯函数：accuracy/f1, ari/nmi, pearson/mse
├── baselines.py     # PCAEmbedder / HVGMeanEmbedder
├── embedder.py      # SpaFMEmbedder（包 SpaFMModel + ckpt）
└── evaluator.py     # 三个 Evaluator + run_benchmark(cfg)
```

---

## 配置示例（[configs/benchmark/spafm-s-eval.yaml](../../configs/benchmark/spafm-s-eval.yaml)）

```yaml
data:
  h5ad_glob: "data/processed/*/demo_*.h5ad"
  tokenizer_config: configs/tokenizer/spafm-s.yaml

embedder:
  type: spafm        # spafm | pca | hvg_mean
  model_config: configs/model/spafm-s.yaml
  ckpt: null         # 留空则随机初始化（仅 smoke）
  batch_size: 8
  device: cpu

tasks:
  - name: cell_type
    label_key: cell_type
    cv_folds: 5
  - name: spatial_domain
    label_key: niche_label
  - name: imputation
    mask_ratio: 0.2

output:
  json_path: runs/bench/spafm-s.json
seed: 42
```

---

## CLI

```bash
python -m scripts.eval_benchmark -c configs/benchmark/spafm-s-eval.yaml
python -m scripts.eval_benchmark -c ... -o embedder.type=pca
```

输出：

- 控制台：每个任务一行结果
- `output.json_path`：结构化结果，便于跨实验对比

---

## 验收清单

- [ ] `make test` 全绿（含 `tests/benchmarks/`）
- [ ] CLI 在 demo h5ad 上跑通三类任务（spafm + pca）
- [ ] 结果 json 写入成功，字段含 `embedder`, `task`, `metric_name`, `value`
- [ ] [README.md](../../README.md) Stage 7 状态 → 🟢 v0

---

## 后续（v1+）

- 接入真实数据集（如 Tabula Sapiens、HEST-1k 子集）
- 多 seed 平均 + 置信区间
- `ScanpyLeiden` baseline（域识别）
- 报告 markdown 自动生成
