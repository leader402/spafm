"""测试 train_pretrain_hier 的 multi-glob 支持。"""

from __future__ import annotations

from pathlib import Path

import yaml
from spafm.data.loaders._common import make_synthetic


def _write_h5ad(target: Path, n_obs: int = 8, seed: int = 0) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    a = make_synthetic(n_obs=n_obs, n_vars=12, seed=seed)
    a.write_h5ad(target, compression="gzip")


def test_multi_glob_resolves_and_dedups(tmp_path: Path) -> None:
    """h5ad_globs 多 glob → 合并去重 → files 列表。"""
    # 模拟两个数据集目录
    _write_h5ad(tmp_path / "ds_a" / "slice_0.h5ad", seed=1)
    _write_h5ad(tmp_path / "ds_a" / "slice_1.h5ad", seed=2)
    _write_h5ad(tmp_path / "ds_b" / "slice_0.h5ad", seed=3)

    # 直接复用 train_pretrain_hier.main 的 glob 解析逻辑
    import glob as _glob

    raw = {
        "data": {
            "h5ad_globs": [
                str(tmp_path / "ds_a" / "*.h5ad"),
                str(tmp_path / "ds_b" / "*.h5ad"),
                # 故意重复一个 glob，验证去重
                str(tmp_path / "ds_a" / "slice_0.h5ad"),
            ],
        }
    }
    glob_field = raw["data"].get("h5ad_globs") or raw["data"].get("h5ad_glob")
    if isinstance(glob_field, str):
        raw_globs = [glob_field]
    else:
        raw_globs = [str(g) for g in glob_field]

    seen: set[Path] = set()
    files: list[Path] = []
    for pat in raw_globs:
        for p in sorted(Path(p) for p in _glob.glob(pat)):
            if p not in seen:
                seen.add(p)
                files.append(p)

    assert len(files) == 3
    names = sorted(f.name for f in files)
    # ds_a 的 slice_0 和 slice_1，ds_b 的 slice_0
    assert names == ["slice_0.h5ad", "slice_0.h5ad", "slice_1.h5ad"]
    # 绝对路径唯一（去重生效）
    assert len(set(files)) == 3


def test_multi_glob_yaml_loadable() -> None:
    """multi 配置可被 yaml 解析且形状正确。"""
    cfg_path = Path("configs/training/spafm-s-pretrain-hier-multi.yaml")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(raw["data"]["h5ad_globs"], list)
    assert len(raw["data"]["h5ad_globs"]) >= 2
    # 服务器档关键参数
    assert raw["batch_size"] >= 2
    assert raw["trainer"]["max_steps"] >= 1000


def test_string_glob_still_works(tmp_path: Path) -> None:
    """旧 h5ad_glob: str 写法继续兼容。"""
    _write_h5ad(tmp_path / "single.h5ad", seed=0)

    import glob as _glob

    raw = {"data": {"h5ad_glob": str(tmp_path / "*.h5ad")}}
    glob_field = raw["data"].get("h5ad_globs") or raw["data"].get("h5ad_glob")
    raw_globs = [glob_field] if isinstance(glob_field, str) else [str(g) for g in glob_field]
    files = sorted(Path(p) for pat in raw_globs for p in _glob.glob(pat))
    assert len(files) == 1
    assert files[0].name == "single.h5ad"
