#!/usr/bin/env bash
# 把服务器跑出来的结果打包成单个 tar.gz，方便 scp 回本地。
#
# 用法：
#     bash scripts/package_results.sh                    # 默认 spafm-results-<date>.tar.gz
#     bash scripts/package_results.sh my-run.tar.gz
#
# 包含：
#     runs/                ← 训练 / 微调 / 评测的全部输出（ckpt + tb 日志）
#     results/             ← benchmark.json / scaling.json 等
#     data/processed/      ← ingest 后的 .h5ad（可选，去掉这一行可大幅瘦身）
#     logs/                ← 任何手动放进来的日志
set -euo pipefail

OUT="${1:-spafm-results-$(date +%Y%m%d-%H%M%S).tar.gz}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INCLUDE=()
for d in runs results logs data/processed; do
    if [ -d "$d" ]; then
        INCLUDE+=("$d")
    fi
done

if [ ${#INCLUDE[@]} -eq 0 ]; then
    echo "✗ 没找到任何要打包的目录（runs/ results/ logs/ data/processed/ 都不存在）"
    exit 1
fi

echo "→ 打包目录: ${INCLUDE[*]}"
tar czf "$OUT" \
    --exclude='**/__pycache__' \
    --exclude='**/.ipynb_checkpoints' \
    --exclude='**/wandb' \
    "${INCLUDE[@]}"

du -h "$OUT" | awk '{print "✓ 完成: " $0}'
echo "回传命令示例:"
echo "  scp $OUT user@local:~/"
