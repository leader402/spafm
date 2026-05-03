.PHONY: help install install-dev lint format test data-demo clean

help:  ## 列出所有可用目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## 安装运行依赖
	pip install -e .

install-dev:  ## 安装开发依赖（lint / test / docs）
	pip install -e ".[dev]"

lint:  ## 代码与 markdown lint
	@echo ">>> Ruff"
	ruff check src tests scripts
	@echo ">>> Black --check"
	black --check --line-length 100 src tests scripts
	@echo ">>> Markdownlint"
	@command -v markdownlint >/dev/null 2>&1 && markdownlint '**/*.md' --ignore node_modules --ignore paper || echo "markdownlint 未安装，跳过"

format:  ## 自动格式化 Python 代码
	ruff check --fix src tests scripts
	black --line-length 100 src tests scripts

test:  ## 运行单元测试
	pytest -q tests

data-demo:  ## 跑通最小数据 pipeline（下载 + 预处理一个 Visium 示例）
	python -m spafm.data.cli build --config configs/data/demo-visium.yaml

clean:  ## 清理临时文件
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build dist *.egg-info
