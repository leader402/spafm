# Post-50K：修复 SVG/对比塌缩 + DLPFC 层聚类下游评测

> 这份是**服务器侧 Codex 的执行手册**。前置：50k-step Hierarchical 预训练已完成，
> ckpt = `runs/spafm-s-hier-multi/ckpt/last-v1.ckpt`。
>
> 本次更新（laptop 端已推到 GitHub）解决了 4 个问题：
>
> 1. SVG 评测 OOM（之前 `max-spots=16` 的 spearman 完全没意义）→ 加分块前向。
> 2. 对比损失 collapse（`loss_ccl → 0.003`）→ view-2 也加独立 mask + 4 个诊断指标。
> 3. DLPFC 12 切片 obs 里的 `layer` 字段为空 → 新合并脚本写入。
> 4. 没有真正可用的 hier ckpt 评测器 → 新增 `HierSpaFMEmbedder`。
>
> 训练 ckpt **不需要重训**就能跑下游；但若想看修好后的对比损失，需要继续训练（见 §5）。

---

## 0. 拉代码

```bash
cd <服务器上的项目根>
git fetch origin
git checkout main
git pull --ff-only origin main
# 验证关键文件已更新
grep -q "chunk_size" src/spafm/benchmarks/svg.py && echo "✓ svg 分块已到位"
grep -q "HierSpaFMEmbedder" src/spafm/benchmarks/embedder.py && echo "✓ hier embedder 已到位"
grep -q "ccl_align" src/spafm/training/hier_lit_module.py && echo "✓ ccl 诊断指标已到位"
ls scripts/data/merge_dlpfc_layer.py && echo "✓ DLPFC 合并脚本已到位"
ls configs/benchmark/spafm-hier-dlpfc.yaml && echo "✓ DLPFC benchmark 配置已到位"
```

---

## 1. 合并 DLPFC 层级真值（< 1 分钟）

```bash
python scripts/data/merge_dlpfc_layer.py \
  --tsv data/raw/spatiallibd-dlpfc-maynard2021/barcode_level_layer_map.tsv \
  --processed-dir data/processed/spatiallibd-dlpfc-maynard2021
```

期望输出 12 行 summary，每行 coverage 接近 90%+。

**自检**：

```bash
python - <<'PY'
import anndata as ad
a = ad.read_h5ad("data/processed/spatiallibd-dlpfc-maynard2021/151673.h5ad")
print("layer counts:", a.obs["layer"].value_counts().to_dict())
print("uns:", a.uns.get("spafm_layer_source"))
PY
```

应见 `L1..L6, WM` 7 类分布。

---

## 2. 重跑 SVG（修好分块后用大 max-spots）

之前 `max-spots=16` 导致 spearman 在 16 个 spot 上几乎全是噪声。现用 `--chunk-size 32 --max-spots 256`：

```bash
mkdir -p results/svg
SAMPLES=(
  "data/processed/spatiallibd-dlpfc-maynard2021/151673.h5ad dlpfc_151673"
  "data/raw/hest-1k-mahmoodlab/st/INT1.h5ad hest_INT1"
  "data/raw/hest-1k-mahmoodlab/st/ZEN1.h5ad hest_ZEN1"
)
for entry in "${SAMPLES[@]}"; do
  read h5 name <<<"$entry"
  echo "=== $name ==="
  spafm eval-svg \
    -h "$h5" \
    --ckpt runs/spafm-s-hier-multi/ckpt/last-v1.ckpt \
    --model-config configs/model/spafm-s-hier.yaml \
    --tokenizer-config configs/tokenizer/spafm-s-visium-mouse.yaml \
    --max-spots 256 --chunk-size 32 --knn 8 --device cuda \
    --out "results/svg/${name}_n256.json" 2>&1 | tail -25
done
```

**判读**：

- `n_spots` 应为 256（不是 16）；
- `spearman_attn_vs_expr` 关心是否高于 baseline `mean_expr_vs_moran_expr`（约 0.2–0.4）；
- 若仍显著低于 baseline，则确认 attention 学到了 housekeeping bias，下一轮训练需要 IDF debias（本次先不改）。

> 显存吃紧再调小：`--chunk-size 16` 或 `--max-spots 128`。

---

## 3. **关键** — DLPFC 层聚类下游评测（新增）

```bash
mkdir -p results
spafm eval -c configs/benchmark/spafm-hier-dlpfc.yaml 2>&1 | tee logs/dlpfc-layer-eval.log
cat results/benchmark-dlpfc-layer.json
```

**关注指标**（应汇报）：

- `spatial_domain.ari` / `spatial_domain.nmi` —— KMeans(k=7) 与 layer 真值的一致度；
- `cell_type.accuracy` / `cell_type.macro_f1` —— 5-fold linear probe；
- 与 baseline 比较：

  ```bash
  # PCA baseline
  cat > /tmp/dlpfc-pca.yaml <<'YAML'
  data:
    h5ad_glob: "data/processed/spatiallibd-dlpfc-maynard2021/*.h5ad"
    tokenizer_config: configs/tokenizer/spafm-s-visium-mouse.yaml
  embedder: { type: pca, n_components: 50 }
  tasks:
    - { name: spatial_domain, label_key: layer }
    - { name: cell_type, label_key: layer, cv_folds: 5 }
  output: { json_path: results/benchmark-dlpfc-layer-pca.json }
  seed: 42
  YAML
  spafm eval -c /tmp/dlpfc-pca.yaml
  ```

**预期**：SpaFM-hier ARI ≥ PCA baseline 才算有效。当前模型若 SVG 仍弱，本任务很可能也低于 PCA → 即可定性"对比塌缩 + housekeeping bias 是真正瓶颈"。

---

## 4. 验证对比塌缩诊断指标已生效（不需重训也能看）

```bash
# 用 dry 训练一两步，看 metrics 是否多了 ccl_pos_sim / ccl_neg_sim / ccl_align / ccl_uniform
python -c "
from spafm.training.hier_lit_module import HierarchicalSpaFMPretrainModule, HierPretrainConfig
cfg = HierPretrainConfig.from_dict({'optim': {'lr':1e-4,'weight_decay':0,'warmup_steps':0,'max_steps':1}})
m = HierarchicalSpaFMPretrainModule(cfg)
print('OK, module loads with new diagnostic logging')
"
```

---

## 5.（可选）继续训练 50k → 100k 看 ccl 是否止跌

修复后两个视图都加独立 mask，理论上 ccl 不会再塌到 0。续训 1k step 即可观察（保留 ckpt 不变，从 last-v1 续起）：

```bash
spafm pretrain -c configs/training/spafm-s-pretrain-hier-multi.yaml \
  --resume runs/spafm-s-hier-multi/ckpt/last-v1.ckpt \
  --override "trainer.max_steps=51000" \
  2>&1 | tee logs/pretrain-resume-1k.log
```

> 实际命令以你们 `spafm pretrain` 的 resume 接口为准；如果没有 `--override`，
> 改 `configs/training/spafm-s-pretrain-hier-multi.yaml` 里的 `max_steps: 51000`。

跑完打开 `runs/spafm-s-hier-multi/lightning_logs/version_X/metrics.csv` 看新出现的列：

- `train/ccl_pos_sim` 应 > `train/ccl_neg_sim`（正样本比负样本相似）；
- `train/ccl_align` 应在 0.1–1.5 之间（不是 ~0）；
- `train/loss_ccl` 应回到 0.5–2.0 量级。

---

## 6. 打包二轮报告

跑完 §1–4 后：

```bash
mkdir -p post50k_v2_report
{
  echo "## DLPFC layer merge"
  python -c "
import anndata as ad, glob
for f in sorted(glob.glob('data/processed/spatiallibd-dlpfc-maynard2021/*.h5ad')):
    a = ad.read_h5ad(f)
    n = a.n_obs
    h = (a.obs['layer'].astype(str) != '').sum() if 'layer' in a.obs else 0
    print(f'{f}: {h}/{n}')
"
  echo "## SVG (n_spots=256)"
  for f in results/svg/*_n256.json; do
    echo "=== $f ==="
    python -c "import json; d=json.load(open('$f')); print({k:d[k] for k in ['n_spots','spearman_attn_vs_expr','top_k_overlap','baseline_spearman']})"
  done
  echo "## DLPFC layer eval"
  cat results/benchmark-dlpfc-layer.json
  echo "## PCA baseline"
  cat results/benchmark-dlpfc-layer-pca.json 2>/dev/null
} > post50k_v2_report/summary.txt

tar -czf post50k_v2_report.tar.gz post50k_v2_report results/svg results/benchmark-dlpfc-layer*.json
sha256sum post50k_v2_report.tar.gz > post50k_v2_report.tar.gz.sha256
git add post50k_v2_report.tar.gz post50k_v2_report.tar.gz.sha256
git commit -m "report: post-50k v2 (svg n=256 + dlpfc layer eval + ccl diagnostics)"
git push
```

---

## 7. 列一遍：服务器需要跑的所有命令

| 步骤 | 是否必跑 | 估时（RTX4090） |
|---|---|---|
| §0 git pull | ✅ | <10s |
| §1 DLPFC layer 合并 | ✅ | <1min |
| §2 SVG n=256 重测（3~5 切片） | ✅ | 3~10min |
| §3 DLPFC layer 聚类 + linear probe | ✅ **核心** | 10~30min（12 切片全过） |
| §3 PCA baseline | ✅ | 1~3min |
| §4 模块加载烟测 | ✅ | <30s |
| §5 续训 1k step 看 ccl | ⚪ 可选 | 10~20min |
| §6 打包报告 | ✅ | <1min |

跑完把 `post50k_v2_report.tar.gz` 推到 GitHub，结束。
