# Codex 执行单：post-50k 修复后的下游评测

> **这是给服务器侧 Codex 的唯一执行文件。** 严格按顺序跑，每步打印关键输出后再走下一步。
> 不要修改任何代码、yaml、配置；不要 stash、不要新 clone；任何步骤报错或异常立即停下回报，**不要自己改逻辑绕过**。

---

## 0. 环境与远端就位

### 0.1 设置 remote（如果没有）

```bash
git remote -v
# 若无 origin，执行：
git remote add origin https://github.com/leader402/spafm.git
# 若已有但 URL 不对，执行：
git remote set-url origin https://github.com/leader402/spafm.git
git fetch origin
```

### 0.2 处理本地未跟踪/改动

按下表处理（**不要 stash、不要 reset、不要新 clone**）：

| 路径 | 处理 |
|---|---|
| `configs/training/spafm-s-pretrain-hier-multi.yaml` | 先 `git diff` 这个文件，把输出贴出来；然后**先不 commit**，直接保留进入 §0.3 的 pull |
| `post50k_report/` | 加进 `.gitignore`，文件原地保留 |
| `results/svg/` | 加进 `.gitignore`，文件原地保留 |
| `runs/`、`logs/`、`data/`、`__pycache__/`、`*.pyc` | 全部忽略，应已被 `.gitignore` 排除；如未排除则补进去 |

执行：

```bash
git diff configs/training/spafm-s-pretrain-hier-multi.yaml | tee /tmp/yaml_diff.txt
grep -qxF 'post50k_report/' .gitignore || echo 'post50k_report/' >> .gitignore
grep -qxF 'results/'        .gitignore || echo 'results/'        >> .gitignore
git status --short
```

### 0.3 拉取远端

```bash
git pull --no-rebase origin main
```

- 若出现冲突 → **立即停下，把冲突文件名和冲突段贴出来**，等指示。
- 若顺利 merge → 继续 §0.4。

### 0.4 自检（5 行都应打印 ✓）

```bash
grep -q "chunk_size" src/spafm/benchmarks/svg.py            && echo "✓ svg 分块"
grep -q "HierSpaFMEmbedder" src/spafm/benchmarks/embedder.py && echo "✓ hier embedder"
grep -q "ccl_align" src/spafm/training/hier_lit_module.py   && echo "✓ ccl 诊断"
ls scripts/data/merge_dlpfc_layer.py >/dev/null && echo "✓ DLPFC 合并脚本"
ls configs/benchmark/spafm-hier-dlpfc.yaml >/dev/null && echo "✓ DLPFC benchmark 配置"
```

✅ **§0 完成的标志**：5 个 ✓ 全部打印 + 把 `git diff` 的 yaml 内容贴出来。
**贴完等回复，不要进入 §1。**

---

## 1. 合并 DLPFC 层级真值

```bash
python scripts/data/merge_dlpfc_layer.py \
  --tsv data/raw/spatiallibd-dlpfc-maynard2021/barcode_level_layer_map.tsv \
  --processed-dir data/processed/spatiallibd-dlpfc-maynard2021
```

### 自检

```bash
python - <<'PY'
import anndata as ad, glob, json
out = []
for f in sorted(glob.glob("data/processed/spatiallibd-dlpfc-maynard2021/*.h5ad")):
    a = ad.read_h5ad(f)
    n = a.n_obs
    h = (a.obs["layer"].astype(str) != "").sum() if "layer" in a.obs else 0
    out.append({"file": f.split("/")[-1], "n_obs": int(n), "n_with_layer": int(h),
                "coverage": round(h / max(1, n), 4)})
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
```

✅ **§1 完成标志**：12 行，每行 coverage ≥ 0.85。把 JSON 输出贴出来。

---

## 2. SVG 重测（修好分块后用 max-spots=256）

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

### 显存兜底
- 若任何一步 OOM → 把命令里 `--chunk-size 32` 改成 `--chunk-size 16` 重跑该样本，**仍 OOM** 再改 `--max-spots 128`。**在最终汇报中写明每个样本实际用的 chunk_size 与 max_spots。**

### 汇报数字（必须给）

```bash
for f in results/svg/*_n256.json; do
  echo "=== $f ==="
  python -c "
import json
d = json.load(open('$f'))
print({
  'n_spots': d['n_spots'],
  'n_genes_scored': d['n_genes_scored'],
  'spearman_attn_vs_expr': round(d['spearman_attn_vs_expr'], 4),
  'top_k_overlap': d['top_k_overlap'],
  'baseline_mean_expr_vs_moran': round(d['baseline_spearman']['mean_expr_vs_moran_expr'], 4),
  'baseline_var_expr_vs_moran':  round(d['baseline_spearman']['var_expr_vs_moran_expr'],  4),
  'top10_attn': d['top_genes_attn'][:10],
})
"
done
```

✅ **§2 完成标志**：每个样本 `n_spots == 256`（或显存兜底后的实际值）。把上面 `for` 输出整段贴出来。

---

## 3. **核心** — DLPFC 层聚类 + linear probe

### 3.1 SpaFM-hier

```bash
mkdir -p logs results
spafm eval -c configs/benchmark/spafm-hier-dlpfc.yaml 2>&1 | tee logs/dlpfc-layer-eval.log
cat results/benchmark-dlpfc-layer.json
```

### 3.2 PCA baseline

```bash
cat > /tmp/dlpfc-pca.yaml <<'YAML'
data:
  h5ad_glob: "data/processed/spatiallibd-dlpfc-maynard2021/*.h5ad"
  tokenizer_config: configs/tokenizer/spafm-s-visium-mouse.yaml
embedder:
  type: pca
  n_components: 50
tasks:
  - name: spatial_domain
    label_key: layer
  - name: cell_type
    label_key: layer
    cv_folds: 5
output:
  json_path: results/benchmark-dlpfc-layer-pca.json
seed: 42
YAML
spafm eval -c /tmp/dlpfc-pca.yaml 2>&1 | tee logs/dlpfc-layer-eval-pca.log
cat results/benchmark-dlpfc-layer-pca.json
```

### 3.3 对比汇报（必须给）

```bash
python - <<'PY'
import json
def pick(p):
    d = json.load(open(p))
    out = {}
    for r in d["results"]:
        out[f"{r['task']}.{r['metric']}"] = round(r["value"], 4)
    return out
hier = pick("results/benchmark-dlpfc-layer.json")
pca  = pick("results/benchmark-dlpfc-layer-pca.json")
keys = sorted(set(hier) | set(pca))
print(f"{'metric':<35} {'hier':>10} {'pca':>10} {'win?':>6}")
for k in keys:
    h, p = hier.get(k, float('nan')), pca.get(k, float('nan'))
    win = "yes" if (isinstance(h, float) and isinstance(p, float) and h > p) else "no"
    print(f"{k:<35} {h:>10} {p:>10} {win:>6}")
PY
```

✅ **§3 完成标志**：把对比表整段贴出来。**明确告诉我：SpaFM-hier 是否在 spatial_domain.ari/nmi 与 cell_type.accuracy/macro_f1 上打过 PCA baseline。**

---

## 4. 烟测：诊断指标已生效

```bash
python - <<'PY'
from spafm.training.hier_lit_module import HierarchicalSpaFMPretrainModule, HierPretrainConfig
cfg = HierPretrainConfig.from_dict({'optim': {'lr':1e-4,'weight_decay':0,'warmup_steps':0,'max_steps':1}})
m = HierarchicalSpaFMPretrainModule(cfg)
print("OK module loads with new diagnostic logging")
PY
```

✅ **§4 完成标志**：打印 `OK module loads ...`。

---

## 5. 续训（**先跳过**，等 §3 结果出来后由用户决定）

不要执行本节，除非用户明确指示。

---

## 6. 打包二轮报告并 push

```bash
mkdir -p post50k_v2_report
{
  echo "## §1 DLPFC layer merge"
  python - <<'PY'
import anndata as ad, glob
for f in sorted(glob.glob("data/processed/spatiallibd-dlpfc-maynard2021/*.h5ad")):
    a = ad.read_h5ad(f)
    n = a.n_obs
    h = (a.obs["layer"].astype(str) != "").sum() if "layer" in a.obs else 0
    print(f"{f}: {h}/{n} = {h/max(1,n):.1%}")
PY

  echo
  echo "## §2 SVG (target n_spots=256)"
  for f in results/svg/*_n256.json; do
    echo "=== $f ==="
    python -c "
import json
d=json.load(open('$f'))
print({
  'n_spots': d['n_spots'],
  'spearman_attn_vs_expr': round(d['spearman_attn_vs_expr'],4),
  'top_k_overlap': d['top_k_overlap'],
  'baseline': {k: round(v,4) for k,v in d['baseline_spearman'].items()},
  'top10_attn': d['top_genes_attn'][:10],
})
"
  done

  echo
  echo "## §3.1 DLPFC layer eval (SpaFM-hier)"
  cat results/benchmark-dlpfc-layer.json

  echo
  echo "## §3.2 DLPFC layer eval (PCA baseline)"
  cat results/benchmark-dlpfc-layer-pca.json

  echo
  echo "## §3.3 head-to-head"
  python - <<'PY'
import json
def pick(p):
    d=json.load(open(p)); o={}
    for r in d["results"]:
        o[f"{r['task']}.{r['metric']}"] = round(r["value"],4)
    return o
hier=pick("results/benchmark-dlpfc-layer.json")
pca =pick("results/benchmark-dlpfc-layer-pca.json")
keys=sorted(set(hier)|set(pca))
print(f"{'metric':<35} {'hier':>10} {'pca':>10} {'win?':>6}")
for k in keys:
    h,p=hier.get(k,float('nan')),pca.get(k,float('nan'))
    win="yes" if (isinstance(h,float) and isinstance(p,float) and h>p) else "no"
    print(f"{k:<35} {h:>10} {p:>10} {win:>6}")
PY
} > post50k_v2_report/summary.txt 2>&1

cp results/benchmark-dlpfc-layer.json     post50k_v2_report/
cp results/benchmark-dlpfc-layer-pca.json post50k_v2_report/
cp -r results/svg                         post50k_v2_report/svg

tar -czf post50k_v2_report.tar.gz post50k_v2_report
sha256sum post50k_v2_report.tar.gz > post50k_v2_report.tar.gz.sha256
ls -lh post50k_v2_report.tar.gz post50k_v2_report.tar.gz.sha256

# 仅添加这两个文件，不要把 results/ post50k_report/ 这些加进 commit
git add post50k_v2_report.tar.gz post50k_v2_report.tar.gz.sha256 .gitignore
git status --short
git commit -m "report: post-50k v2 (svg n=256 + dlpfc layer eval vs pca + ccl diagnostics)"
git push origin main
```

✅ **§6 完成标志**：`git push` 成功，把最后 5 行 push 输出贴出来。

---

## 失败/异常处理（关键）

| 现象 | 动作 |
|---|---|
| `git pull` 冲突 | 停，贴冲突文件和段，等指示 |
| `python scripts/data/merge_dlpfc_layer.py` 报 `KeyError`/缺文件 | 停，贴 stderr |
| `spafm eval-svg` CUDA OOM | 按 §2 兜底降 chunk_size→16，再降 max_spots→128；记录实际值 |
| `spafm eval` 找不到 ckpt | 停，`ls -la runs/spafm-s-hier-multi/ckpt/` 贴出来 |
| `spafm eval` 找不到 `obs['layer']` | 说明 §1 没跑；回去重跑 §1 |
| `git push` 被拒 | 先 `git pull --no-rebase origin main`，再推；冲突就停下贴出来 |
| 任何一步 traceback | **停下，整段贴出来**；不要修复代码，不要忽略 |

---

## 最终汇报模板（跑完所有步骤后给用户）

```
=== §0 自检 ===
（5 行 ✓）

=== §0.2 yaml diff ===
（git diff 输出）

=== §1 DLPFC layer coverage ===
（12 行 JSON）

=== §2 SVG ===
（3 个样本的指标，标明每个实际用的 chunk_size / max_spots）

=== §3 head-to-head ===
（hier vs pca 表）
SpaFM-hier 打过 PCA baseline 的指标：xxx
SpaFM-hier 输给 PCA baseline 的指标：xxx

=== §4 烟测 ===
OK module loads ...

=== §6 push ===
commit: <hash>  remote: origin/main
artifact: post50k_v2_report.tar.gz （xx KB, sha256=xxxx）
```
