# 服务器迁移与启动手册（server-launch）

> 配套：[server-data-prep.md](server-data-prep.md)（先把数据准备好），本手册关心**代码迁移 + 训练启动**。
> 目标：把笔记本上跑通的 Hierarchical SpaFM 平移到 GPU 服务器（单卡 / 多卡 DDP）。

---

## 1. 前置清单

| 项 | 要求 |
|---|---|
| OS | Linux（Ubuntu 22.04 / CentOS 7+ 验证过） |
| GPU | ≥ 1 张 ≥ 24GB；推荐 A100/H100/L40s/4090 |
| CUDA | 12.x（与 PyTorch 2.2+ 匹配） |
| Python | 3.10 或 3.12 |
| 磁盘 | 数据 ≥ 200 GB（A 档 ~30 GB / B 档 1.5 TB） |
| 网络 | 能访问 huggingface.co、10x CDN、s3.amazonaws.com |

---

## 2. 代码迁移（两种方式择一）

### 方式 A：git clone（推荐）

```bash
# 服务器上
cd ~/projects
git clone <your-repo-url> ST-FoundationModel
cd ST-FoundationModel
git checkout main   # 或具体 tag
```

### 方式 B：rsync（无 git remote 时）

```bash
# 笔记本上
rsync -avzP --exclude '.venv' --exclude 'data' --exclude 'runs' \
  --exclude '__pycache__' --exclude '.git/objects' \
  ./ user@server:~/projects/ST-FoundationModel/
```

> ⚠️ **不要传 [data/](data/) 与 [runs/](runs/)**：前者用 `scripts/data/` 在服务器本地下载，后者是产物。

---

## 3. 环境构建

```bash
# 1. mamba 安装基础环境
mamba env create -f environment.yml -n spafm
conda activate spafm

# 2. 可编辑安装
pip install -e ".[dev]"

# 3. 验证
python -c "import spafm, torch; print(spafm.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

期望输出形如：`0.x.x True 4`。

---

## 4. 数据准备

按 [server-data-prep.md](server-data-prep.md) 执行 A 档（≈30 切片，1–2 小时），落到 `data/processed/`：

```text
data/processed/
├── visium-mouse-brain-sagittal-anterior/V1_Mouse_Brain_Sagittal_Anterior.h5ad
├── visium-mouse-brain-sagittal-posterior/...
├── visium-mouse-brain-coronal/...
├── visium-mouse-kidney/...
├── visium-human-breast-cancer-block-a/...
├── visium-human-heart/...
├── visium-human-lymph-node/...
├── visium-human-brain-cancer/...
├── visium-human-cerebellum/...
└── spatiallibd-dlpfc-maynard2021/151{507..676}/*.h5ad
```

快速校验：

```bash
ls data/processed/**/*.h5ad | wc -l   # ≥ 25 即可启动
```

---

## 5. 单卡 smoke 测试（必跑，5–10 分钟）

```bash
# 用本地小配置确认链路
spafm pretrain-hier -c configs/training/spafm-s-pretrain-hier-mouse.yaml \
  trainer.max_steps=20 trainer.log_every_n_steps=5
```

通过判据：

- 不报 OOM；
- `train/mgm_loss` 单调下降；
- `runs/.../checkpoints/` 出现 `last.ckpt`。

---

## 6. 多卡 DDP 启动

### 6.1 修改 trainer 字段

直接 CLI override（推荐，不污染 YAML）：

```bash
# 4 卡 DDP
spafm pretrain-hier -c configs/training/spafm-s-pretrain-hier-multi.yaml \
  trainer.devices=4 trainer.strategy=ddp \
  trainer.precision=bf16-mixed \
  batch_size=4 num_workers=8
```

> Lightning 在 `devices > 1` 时会自动用 `torchrun`。**全局 batch = `batch_size × devices`**。学习率不必手动放大（已用 warmup 缓冲）。

### 6.2 8 卡 + 梯度累积（更大 effective batch）

```bash
spafm pretrain-hier -c configs/training/spafm-s-pretrain-hier-multi.yaml \
  trainer.devices=8 trainer.strategy=ddp \
  trainer.accumulate_grad_batches=2 \
  batch_size=4
# effective batch = 4 × 8 × 2 = 64
```

### 6.3 后台运行 + 日志

```bash
mkdir -p logs
nohup spafm pretrain-hier -c configs/training/spafm-s-pretrain-hier-multi.yaml \
  trainer.devices=4 trainer.strategy=ddp \
  > logs/spafm-s-multi-$(date +%Y%m%d-%H%M%S).log 2>&1 &
echo $! > logs/last.pid
tail -f logs/spafm-s-multi-*.log
```

---

## 7. Checkpoint 续训

```bash
# 自动找 last.ckpt
spafm pretrain-hier -c configs/training/spafm-s-pretrain-hier-multi.yaml \
  trainer.devices=4 trainer.strategy=ddp \
  ckpt_path=runs/spafm-s-hier-multi/lightning_logs/version_0/checkpoints/last.ckpt
```

> 若 `ckpt_path` 字段当前未在 `pretrain-hier` CLI 暴露，请改 [configs/training/spafm-s-pretrain-hier-multi.yaml](../../configs/training/spafm-s-pretrain-hier-multi.yaml) 顶层加 `ckpt_path: <path>`。**TODO: 验证 CLI 是否已支持**。

---

## 8. 监控（可选但推荐）

### TensorBoard（默认）

```bash
tensorboard --logdir runs/ --port 6006 --bind_all
# SSH tunnel： ssh -L 6006:localhost:6006 user@server
```

### Weights & Biases

```bash
export WANDB_API_KEY=...        # 不要写进仓库
export WANDB_PROJECT=spafm-pretrain
spafm pretrain-hier -c configs/training/spafm-s-pretrain-hier-multi.yaml \
  trainer.devices=4 logger=wandb
```

> **TODO: `logger=wandb` 是否已在 CLI 暴露**——若否，需先扩展 [src/spafm/training/cli.py](../../src/spafm/training/cli.py)。

---

## 9. 常见坑

| 现象 | 原因 | 修复 |
|---|---|---|
| `NCCL error: unhandled cuda error` | 多卡 IB / NVLink 异常 | `export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` 后重试 |
| DDP 卡在 init | `MASTER_PORT` 端口冲突 | `export MASTER_PORT=29511`（任选闲置端口） |
| OOM at step 0 | `n_spots_per_sample × samples_per_slice` 过大 | 先降到 `N=64, samples=16` |
| `num_workers` 把内存吃满 | 多 slice + h5ad 缓存 | `num_workers=4`，必要时 `persistent_workers=true` |
| 数据加载比训练慢 | h5ad I/O 瓶颈 | 开 `samples_per_slice` 加大单次复用；或改用 mmap |
| HF 下载 401 | HEST 需 gated access | `huggingface-cli login` 后再跑 [scripts/data/download_hest.py](../../scripts/data/download_hest.py) |

---

## 10. 提交前清单

- [ ] `make lint` / `make test` 全绿
- [ ] 训练日志归档到 `runs/<run-name>/log.txt`
- [ ] 关键超参与配置文件名记录到 [docs/playbooks/stage8-工程化.md](stage8-工程化.md) 的实验台账
- [ ] checkpoint 用 `last.ckpt` + `epoch=*-step=*.ckpt` 双备份
- [ ] 数据来源与引用：参见 [docs/references/数据集引用.md](../references/数据集引用.md)

---

## 11. 一行启动备忘

```bash
# A 档 4 卡，BF16，DDP
nohup spafm pretrain-hier -c configs/training/spafm-s-pretrain-hier-multi.yaml \
  trainer.devices=4 trainer.strategy=ddp trainer.precision=bf16-mixed \
  > logs/run-$(date +%m%d-%H%M).log 2>&1 &
```
