# Stage 6 — 下游适配（LoRA / Linear Probe / Fine-tune）

> 状态：🟢 v0 完成（三种适配策略 + 三种下游 head + Lightning 微调模块）
>
> 上游：[stage5-知识增强.md](stage5-知识增强.md)　下游：[stage7-评测框架.md](stage7-评测框架.md)

---

## 1. 目标

把 Stage 4/5 预训练好的 `SpaFMModel` 适配到具体下游任务，证明"预训练 → 下游"链路完整。v0 在笔记本上跑通三种适配范式 × 三种任务的"形状级"组合，真实大规模微调留给服务器。

| 适配策略 | 可训练参数 | 适用场景 |
|---|---|---|
| **Linear Probe** | 仅下游 head | 评估表征质量；最便宜 |
| **LoRA** | head + LoRA A/B 矩阵 | 资源中等；保持主干冻结避免遗忘 |
| **Full Fine-tune** | 全部 | 资源充足、目标任务分布偏移大 |

| 下游任务 | head | label 形态 |
|---|---|---|
| 细胞类型注释 | `CellTypeHead` (cell_repr → C) | `obs["cell_type"]` 整数 |
| 空间域识别 | `SpatialDomainHead`（同上，类别少） | `obs["domain"]` 整数 |
| 基因表达插补 | `ImputationHead` (token_repr → 1) | mask 后预测原 `value_floats` |

---

## 2. LoRA 实现

对 nn.Linear 注入低秩适配：

$$
y = Wx + \frac{\alpha}{r}\, B (A x)
$$

其中 $A \in \mathbb{R}^{r\times d_\text{in}}$、$B \in \mathbb{R}^{d_\text{out}\times r}$，$W$ 冻结。
v0 默认对 attention 的 q/k/v/out 投影注入（`target_modules=["q","k","v","out"]`）。

工具：

- [adaptation/lora.py](../../src/spafm/adaptation/lora.py)
  - `LoRALinear(base: nn.Linear, r, alpha, dropout)` 包装一个已存在的 Linear
  - `apply_lora(model, r, alpha, target_modules)` 递归替换匹配名字的 Linear
  - `mark_only_lora_as_trainable(model, train_head=True)` 冻结其他参数

---

## 3. 模块边界

| 模块 | 职责 |
|---|---|
| [adaptation/lora.py](../../src/spafm/adaptation/lora.py) | LoRALinear、`apply_lora`、`mark_only_lora_as_trainable`、`count_trainable` |
| [adaptation/heads.py](../../src/spafm/adaptation/heads.py) | `CellTypeHead`、`SpatialDomainHead`、`ImputationHead` |
| [adaptation/dataset.py](../../src/spafm/adaptation/dataset.py) | `LabeledH5ADDataset`：在 H5ADCorpusDataset 基础上同时返回 label |
| [adaptation/lit_module.py](../../src/spafm/adaptation/lit_module.py) | `SpaFMFinetuneModule`：装载预训练 ckpt + 适配策略 + head |
| [scripts/train_finetune.py](../../scripts/train_finetune.py) | CLI 入口 |

---

## 4. 配置示例

[configs/training/spafm-s-finetune.yaml](../../configs/training/spafm-s-finetune.yaml)：

```yaml
data:
  h5ad_glob: "data/processed/*/demo_*.h5ad"
  tokenizer_config: configs/tokenizer/spafm-s.yaml
  label_key: cell_type           # adata.obs[label_key] 必须存在
  task: classification           # classification | imputation

model_config: configs/model/spafm-s.yaml
pretrained_ckpt: null            # 可选：加载 Stage4/5 ckpt

adaptation:
  strategy: lora                 # linear_probe | lora | full
  lora:
    r: 8
    alpha: 16
    dropout: 0.0
    target_modules: [q_proj, k_proj, v_proj, out_proj]

head:
  type: cell_type                # cell_type | spatial_domain | imputation
  num_classes: 8                 # 仅分类任务
  hidden: 128

optim:
  lr: 1.0e-3
  weight_decay: 0.01
  warmup_steps: 20
  max_steps: 200

trainer:
  accelerator: auto
  devices: 1
  precision: 32
  max_steps: 200
  log_every_n_steps: 10
  default_root_dir: runs/spafm-s-ft

batch_size: 8
num_workers: 0
seed: 42
```

---

## 5. 使用流程

```bash
# 1. 仅 head（最便宜）
python -m scripts.train_finetune -c configs/training/spafm-s-finetune.yaml \
  -o adaptation.strategy=linear_probe

# 2. LoRA 微调（默认）
python -m scripts.train_finetune -c configs/training/spafm-s-finetune.yaml

# 3. 全量微调
python -m scripts.train_finetune -c configs/training/spafm-s-finetune.yaml \
  -o adaptation.strategy=full -o optim.lr=3.0e-5
```

---

## 6. 验收清单

- [x] `LoRALinear` 数学正确：$\Delta W = (\alpha/r)\,BA$，$B$ 初始化为 0 → 加载即等价 $W$
- [x] `apply_lora` 按名字匹配替换 attention 的 q/k/v/out
- [x] `mark_only_lora_as_trainable` 三种策略下可训练参数数量正确
- [x] 三种 head 形状/loss 正确
- [x] `SpaFMFinetuneModule` 端到端 fit 不报错，loss 下降
- [x] CLI smoke：CPU 5 步跑通三种策略
- [ ] 真实下游基准（Stage 7）

---

## 7. 关键 trick

- **q/k/v/out 命名**：[src/spafm/models/attention.py](../../src/spafm/models/attention.py) 的 Linear 名字必须能被 target_modules 匹配 → 这次顺手把内部命名规范化为 `q_proj/k_proj/v_proj/out_proj`
- **head pooling**：`cell_repr = h[:, 0, :]`（CLS）即可，无需 attention pooling
- **加载 ckpt**：用 `torch.load(weights_only=False)` 后 `model.load_state_dict(strict=False)`，允许 head 参数缺失
