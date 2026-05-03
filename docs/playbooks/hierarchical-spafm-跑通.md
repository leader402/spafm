# Hierarchical SpaFM 笔记本跑通手册

> 状态：✅ 已在 RTX 5080 Laptop (16GB) + WSL2 跑通预训练全流程
> 提交：本 PR
> 关联：`docs/playbooks/laptop-v0-真实数据跑通.md`（v0 per-cell 版本）

## 1. 为什么需要这个重构

### v0 的隐藏 bug

v0（per-cell-only `SpaFMModel`）中，每个 cell 内部所有基因 token **共享同一个 `coord`**。
`SpatialMHSA.spatial_bias = -alpha * dist2 / sigma2`，其中 `dist2` 由 token 的 coord 计算 ——
所以 v0 的 `dist2` 在每个 cell 内恒为 0，**spatial bias 形同虚设**。

实测 v0 真实数据：

```text
coords[0, :5] = (8500, 7474), (8500, 7474), (8500, 7474), ...  # 同一 cell 全部相同
dist2.max() = 0.0
```

这意味着 v0 的"空间感知注意力"是个**假创新**。论文不能这样写。

### v1 (Hierarchical) 的修复

把模型分成两层：

| 层 | 输入 | 注意力 | 用途 |
|---|---|---|---|
| inner | `(B*N_spots, L_genes)` 基因序列 | gene-gene（每个 spot 内部） | 复用原 `SpaFMModel` |
| outer | `(B, N_spots, d)` spot 序列 | **spot-spot（带真实 spot 坐标）** | 新增 `outer_blocks` |

外层每个 token 是一个 spot 的 `cell_repr`，token 间距离用真实 Visium spot 坐标计算 ——
**这才是真正的 spatial-aware attention**。

实测 v1 真实数据（2 slice × 24 spots）：

```text
spot_coords shape: (2, 24, 2)
slice 0: dist2 max=62,796,920, mean=13,788,822, #pairs>0 = 552/576 (96%)
slice 1: dist2 max=51,314,244, mean=11,749,681, #pairs>0 = 552/576
outer attn shape: (2, 4, 24, 24)
row diversity (max-min mean): 0.4331  ✅ 远大于均匀 1/24 = 0.042
```

## 2. 笔记本跑通配置

```bash
spafm pretrain-hier -c configs/training/spafm-s-pretrain-hier-mouse.yaml
```

关键参数（已写入 yaml 默认）：

- `data.n_spots_per_sample=24`、`data.samples_per_slice=16`
- `batch_size=2` → 每 step 48 cells
- `trainer.precision=bf16-mixed`
- 模型：内层 6 层 + 外层 2 层 = 25.1M 参数

笔记本资源消耗：

- GPU 显存峰值：**15.9 GB / 16 GB**（接近上限但稳定）
- 训练速度：约 **3.7 s/step**（100 步 ≈ 6 分钟，但首轮 epoch 数据加载较慢，实测 100 步 ~36 分钟）
- 系统内存：约 5 GB

## 3. 100 步收敛证据

| step | total | mgm | ccl |
|---:|---:|---:|---:|
| 0 | 11.16 | 11.05 | 1.05 |
| 100 | **7.96** | **7.89** | **0.70** |

强收敛，loss 下降 28.7%。

## 4. 服务器迁移步骤

代码与配置已就绪，迁移到 ≥40GB 显存的服务器：

1. `git pull` + `pip install -e .`
2. 复制 `data/processed/` 与 `data/external/`
3. 修改 `configs/training/spafm-s-pretrain-hier-mouse.yaml`：
    - `n_spots_per_sample: 128`（4-8 倍）
    - `samples_per_slice: 32`
    - `batch_size: 4-8`
    - `optim.max_steps: 50000+`
    - `trainer.devices: 多卡`（按需）
4. `spafm pretrain-hier -c configs/training/spafm-s-pretrain-hier-mouse.yaml`

## 5. 测试覆盖

[tests/training/test_hier_pretrain.py](../../tests/training/test_hier_pretrain.py)：

- `test_slice_dataset_and_collator` — 数据形状
- `test_outer_spatial_bias_active_on_real_coords` — **守门测试**：断言 `dist2 > 0`、`attn diversity > 1e-3`
- `test_hier_pretrain_module_one_step` — 端到端训练 1 步 + 梯度有限

全仓库测试：**123 passed**（v0 的 120 + 新增 3）。

## 6. 下一步

- 迁服务器跑长程训练（≥10K 步）
- T-B：基于 outer attention 做细胞间通讯（CCC）下游任务
- T-D：基于 inner attention 做空间可变基因（SVG）下游任务
