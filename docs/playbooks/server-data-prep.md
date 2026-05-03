# 服务器侧数据准备 Playbook

> 状态：📋 操作手册（笔记本侧脚本已就绪，服务器执行待验）
> 配套：[scripts/data/download_visium.py](../../scripts/data/download_visium.py)、
> [scripts/data/download_spatiallibd.py](../../scripts/data/download_spatiallibd.py)、
> [scripts/data/download_hest.py](../../scripts/data/download_hest.py)

## 1. 目标分层

按服务器算力 / 时间预算分三档：

| 档位 | 切片数 | 大小 | 时间 | 用途 |
|---|---|---|---|---|
| **A — MVP** | 30~40 | ~10GB | 半天 | preprint / workshop |
| **B — 论文主结果** | 1100+ | ~1TB | 1-2 月 | 投会 / 期刊 |
| **C — scaling** | 5K+ | 5TB+ | 3+ 月 | 跨平台融合 |

当前推荐：**A → B 渐进**，先把 A 跑通再扩 B。

---

## 2. 档位 A：30 切片（半天搞定）

### 2.1 10x Visium 全部 20 个直链切片

```bash
cd <repo_root>
python scripts/data/download_visium.py --list           # 看清单
python scripts/data/download_visium.py --all            # 全量拉
```

预计：~2GB，~30 分钟（视带宽）。

包含组织：鼠脑（sagittal anterior/posterior + 各 s2 + coronal × 3）、
鼠肾、人乳腺（×2）、人心、人淋巴结、人脑（cerebellum/glioblastoma/spinal cord/brain cancer）、
人卵巢/结直肠/肺癌（FFPE）。

### 2.2 spatialLIBD DLPFC 12 切片（评测黄金基准）

```bash
python scripts/data/download_spatiallibd.py --list
python scripts/data/download_spatiallibd.py --all
```

预计：~5GB。这 12 切片自带 layer 标注，是 niche/domain 评测的金标准
（STAGATE/GraphST/SpaGCN 论文都用它）。

> ⚠️ 当前脚本里的 `BASE_URL` 是 spatialLIBD 公开镜像位置，
> 服务器上首跑前**先用 `--sample 151673` 测一个**，确认 URL 可用；
> 若 LIBD 实验室更新了路径，按提示修正 [scripts/data/download_spatiallibd.py](../../scripts/data/download_spatiallibd.py) 的 `BASE_URL`。

### 2.3 验收

```bash
ls -lh data/raw/                          # 应有 ~22 个目录
du -sh data/raw/                          # 总大小约 8-10GB
spafm pretrain-hier -c configs/training/spafm-s-pretrain-hier-mouse.yaml \
    -o trainer.max_steps=20 -o data.samples_per_slice=4    # 跑通 smoke
```

---

## 3. 档位 B：HEST-1k（1108 切片，~1TB）

### 3.1 准备工作

```bash
# 1) 安装依赖
pip install huggingface_hub pandas

# 2) HuggingFace 认证（必须，HEST-1k 需要同意 license）
huggingface-cli login   # 浏览器拿 token 粘进来

# 3) 同意 HEST-1k 的 license
#    访问 https://huggingface.co/datasets/MahmoodLab/hest 点 Agree

# 4) 准备 ≥ 1.5TB 磁盘
df -h /path/to/data
```

### 3.2 探查 metadata（轻量，~10MB）

```bash
python scripts/data/download_hest.py --metadata-only
```

会打印 `species × st_technology` 透视表 + Top-15 organ 分布。
**先看清楚再决定下载子集。**

### 3.3 分阶段下载

**阶段 1：小子集冒烟（~5GB，验证流程）**

```bash
python scripts/data/download_hest.py \
    --species human --tissue brain --platform visium \
    --max-samples 5
```

**阶段 2：单平台中等子集（~100GB）**

```bash
# Visium 全部（约 600 切片）
python scripts/data/download_hest.py --platform visium --workers 8
```

**阶段 3：全量（~1TB）**

```bash
nohup python scripts/data/download_hest.py --all --workers 8 \
    > logs/hest_download.log 2>&1 &
```

预计 1-2 天（10 Gbit 带宽）。

### 3.4 数据组织

下载后结构：

```
data/raw/hest-1k-mahmoodlab/
├── HEST_v1_1_0.csv             # metadata
├── st/<id>.h5ad                # 表达 + 空间
├── thumbnails/<id>.jpg         # H&E 缩略图
└── metadata/<id>.json
```

每个 `.h5ad` 已经做过 QC、对齐了 H&E 坐标，
可直接被 [src/spafm/data/loaders/](../../src/spafm/data/loaders/) 消费
（HEST loader 待补 — TODO Stage 8）。

---

## 4. 国内镜像 / 加速

10x cf.10xgenomics.com 在国内偶尔慢。可选方案：

1. **走代理**：服务器上配 `http_proxy` / `https_proxy`
2. **挂阿里云镜像**（如有）：在 [download_visium.py](../../scripts/data/download_visium.py) 的 `MIRRORS["cn"]` 里填
3. **HuggingFace 加速**：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 5. 训练侧改动（数据扩充后）

完成 A 档后，需要更新 hier 预训练 config：

```yaml
# configs/training/spafm-s-pretrain-hier-multi.yaml（新建）
data:
  slice_globs:
    - data/processed/visium-mouse-brain-*/V1_*.h5ad
    - data/processed/visium-human-*/V1_*.h5ad
    - data/processed/spatiallibd-dlpfc-maynard2021/*/V1_*.h5ad
  samples_per_slice: 32
  max_spots: 256
trainer:
  max_steps: 50000
  precision: bf16-mixed
optim:
  lr: 1e-4
  warmup_steps: 1000
```

> Loader/dataset 侧若还不支持 multi-slice glob，按 [docs/playbooks/hierarchical-spafm-跑通.md](hierarchical-spafm-跑通.md) §5 扩展。

---

## 6. 验收 checklist

A 档完成后应能：

- [ ] `ls data/raw/ | wc -l` ≥ 22
- [ ] `spafm pretrain-hier ... max_steps=20` 在多 slice 上跑通
- [ ] `spafm eval-svg --h5ad data/processed/spatiallibd-dlpfc-maynard2021/151673/...` 跑通
- [ ] HEST metadata 可成功打印分布

---

## 7. 后续 / TODO

- [ ] 写 `src/spafm/data/loaders/hest.py`（统一 h5ad → SpaFM 输入）
- [ ] 写 `src/spafm/data/loaders/spatiallibd.py`（含 layer 标注解析）
- [ ] CI 加 `test_data_smoke.py`：跑通 `--metadata-only`（不下大文件）
- [ ] STOmics / SODB 第三方镜像探查
