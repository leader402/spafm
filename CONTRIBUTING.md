# 贡献指南 — SpaFM

感谢您愿意为 SpaFM 做出贡献！

## 工作流

1. **先读文档**：`README.md` → 当前阶段的 `docs/playbooks/stageX-*.md`
2. **建分支**：`git checkout -b feat/<scope>-<short-desc>`
3. **小步提交**：单个 PR 聚焦一件事
4. **跑门禁**：`make lint && make test`
5. **同步文档**：功能、配置、命令变更需同步更新 README / docs / configs

## Commit 规范

```
feat|fix|docs|chore|refactor|test|perf: scope - summary
```

## 代码风格

- Python 3.10+，Black (line=100) + Ruff
- 公共 API 必须有 docstring + type hint
- 文档中文，代码英文

## 数据贡献

- 新增数据集必须登记到 [docs/references/数据集清单.md](docs/references/数据集清单.md)
- 不直接 commit 原始数据；改为提供下载脚本 `scripts/data/download_<source>.py`

## 报告问题

提交 Issue 时请包含：
- 复现步骤
- 期望行为 vs 实际行为
- 环境信息（`python -V`、`pip freeze | grep -E "torch|scanpy|anndata"`）
