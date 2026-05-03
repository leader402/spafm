"""Stage 9 scaling 层测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from scripts.scaling_report import app
from spafm.models import ModelConfig, SpaFMModel
from spafm.scaling import (
    SIZE_CONFIGS,
    count_params,
    estimate_flops_per_token,
    estimate_params_from_cfg,
    fit_scaling_law,
    get_size_config,
)

runner = CliRunner()


# --------------------------------------------------------------------------- #
def test_size_configs_all_loadable() -> None:
    for s in ("S", "M", "L"):
        cfg = get_size_config(s)
        assert cfg.vocab_size > 0 and cfg.d_model > 0


def test_get_size_unknown() -> None:
    with pytest.raises(KeyError):
        get_size_config("XL")


def test_yaml_configs_match_sizes() -> None:
    """configs/model/spafm-{s,m,l}.yaml 与 SIZE_CONFIGS 的核心字段一致。"""
    for s, fname in [("S", "spafm-s.yaml"), ("M", "spafm-m.yaml"), ("L", "spafm-l.yaml")]:
        from_yaml = ModelConfig.from_yaml(f"configs/model/{fname}")
        from_mem = SIZE_CONFIGS[s]
        for k in ("d_model", "n_layers", "n_heads", "d_ffn", "vocab_size"):
            assert getattr(from_yaml, k) == getattr(from_mem, k), f"{s} 档 {k} 不一致"


def test_estimate_vs_measured_S() -> None:
    """估计值与真实参数误差 < 20%。"""
    cfg = ModelConfig(vocab_size=2000, d_model=64, n_layers=2, n_heads=4, d_ffn=128, d_pos=32)
    est = estimate_params_from_cfg(cfg)["total"]
    real = count_params(SpaFMModel(cfg))
    rel = abs(est - real) / real
    assert rel < 0.2, f"估计 {est} vs 真实 {real}，相对误差 {rel:.2%}"


def test_flops_proportional_to_params() -> None:
    cfg = ModelConfig(vocab_size=2000, d_model=64, n_layers=2, n_heads=4, d_ffn=128, d_pos=32)
    blocks = estimate_params_from_cfg(cfg)["blocks"]
    assert estimate_flops_per_token(cfg) == pytest.approx(6.0 * blocks)


# --------------------------------------------------------------------------- #
def test_fit_scaling_law_recovers_alpha() -> None:
    rng = np.random.default_rng(0)
    alpha_true, A_true = 0.34, 100.0
    Ps = np.logspace(7, 10, 8)
    Ls = A_true * Ps ** (-alpha_true) * (1 + rng.normal(0, 0.005, size=Ps.size))
    fit = fit_scaling_law(list(zip(Ps.tolist(), Ls.tolist(), strict=True)))
    assert abs(fit.alpha - alpha_true) / alpha_true < 0.05
    assert fit.r2 > 0.99


def test_fit_scaling_law_too_few() -> None:
    with pytest.raises(ValueError):
        fit_scaling_law([(1e7, 1.0)])


# --------------------------------------------------------------------------- #
def test_cli_scaling_report_basic() -> None:
    res = runner.invoke(app, [])
    assert res.exit_code == 0
    for s in ("S", "M", "L"):
        assert s in res.stdout


def test_cli_scaling_report_with_fit(tmp_path: Path) -> None:
    pts = [{"params": p, "loss": 100.0 * p ** (-0.3)} for p in (1e7, 5e7, 2e8, 1e9)]
    fp = tmp_path / "pts.json"
    fp.write_text(json.dumps(pts), encoding="utf-8")
    res = runner.invoke(app, ["--fit", str(fp)])
    assert res.exit_code == 0
    assert "alpha" in res.stdout
