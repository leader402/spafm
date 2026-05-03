# Stage 9 · SpaFM-L 扩展与 Scaling Law（v0）

## 目标

为 SpaFM 提供**多档模型规模**与**扩展规律分析**工具，回答两类问题：

1. 同一份 codebase 在 S → M → L 三档的参数 / FLOPs / 显存预算如何？
2. 给定多个 (params, loss) 实测点，按 Chinchilla/Hoffmann 范式做最小二乘拟合，
   给出指数 ``a`` 与基线 ``E``。

> v0 不做真实大规模训练（无算力）；提供**配置 + 工具 + 报告**。

---

## 三档配置

| 档 | d_model | n_layers | n_heads | d_ffn | 参数量（粗估） |
|---|---|---|---|---|---|
| **S** | 256 | 6  | 4  | 1024 | ~10 M |
| **M** | 512 | 12 | 8  | 2048 | ~80 M |
| **L** | 1024 | 24 | 16 | 4096 | ~600 M |

对应文件：

- [configs/model/spafm-s.yaml](../../configs/model/spafm-s.yaml)（已存在）
- [configs/model/spafm-m.yaml](../../configs/model/spafm-m.yaml)
- [configs/model/spafm-l.yaml](../../configs/model/spafm-l.yaml)

---

## 模块布局

```
src/spafm/scaling/
├── __init__.py
├── sizes.py          # SIZE_CONFIGS dict（与 yaml 对齐的内存版）
├── params.py         # count_params(model) + estimate_params_from_cfg(cfg)
├── flops.py          # estimate_flops_per_token(cfg)
└── scaling_law.py    # fit_scaling_law(points) → (a, E)
```

---

## 关键公式

**参数估计**（忽略 LayerNorm / bias 等小项）：

$$
P \approx V d + L \cdot (4 d^2 + 2 d \cdot d_\text{ffn})
$$

其中 $V$=vocab_size, $d$=d_model, $L$=n_layers, $d_\text{ffn}$=ffn 中间维。

**FLOPs/token**（Transformer 一次前向 + 反向 ≈ 6P）：

$$
F \approx 6 P
$$

**Scaling-law（loss 关于参数）**：

$$
L(P) = E + \frac{A}{P^{\alpha}}
$$

v0 给固定 $E=0$ 的简化版：$\log L = \log A - \alpha \log P$ → 一次最小二乘。

---

## CLI

```bash
python -m scripts.scaling_report                       # 打印 S/M/L 三档
python -m scripts.scaling_report --fit results.json    # 拟合 scaling law
```

`results.json` 格式：

```json
[{"params": 1.0e7, "loss": 4.2}, {"params": 8.0e7, "loss": 3.5}, ...]
```

---

## 验收清单

- [x] 三档 yaml 存在且可被 `ModelConfig.from_yaml` 加载
- [x] `estimate_params_from_cfg` 与真实 `count_params` 误差 < 20%
- [x] `fit_scaling_law` 在合成数据上恢复指数（误差 < 5%）
- [x] CLI 打印 S/M/L 报告；`make test` 全绿
- [x] [README.md](../../README.md) Stage 9 → 🟢 v0

---

## 后续（v1+）

- 真实多档预训练实验 + 数据点采集
- 接入 `torch.utils.flop_counter` / `fvcore` 做精确 FLOPs
- 拟合带常数项 $E$ 的完整三参数模型
- 给出"算力预算 → 推荐 (params, tokens)"曲线
