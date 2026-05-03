"""Stage 8 工程化层测试：registry + 顶层 CLI smoke。"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from spafm import __version__
from spafm.cli import app
from spafm.registry import MODEL_REGISTRY, ModelCard, get_model_card, list_models

runner = CliRunner()


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_has_default() -> None:
    assert "spafm-s-v0" in MODEL_REGISTRY
    cards = list_models()
    assert all(isinstance(c, ModelCard) for c in cards)
    assert any(c.id == "spafm-s-v0" for c in cards)


def test_registry_get_unknown() -> None:
    with pytest.raises(KeyError):
        get_model_card("does-not-exist")


def test_modelcard_status_placeholder() -> None:
    c = get_model_card("spafm-s-v0")
    assert c.status == "placeholder"
    assert c.license == "MIT"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_help() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for sub in ("version", "data", "model", "pretrain", "finetune", "eval"):
        assert sub in res.stdout


def test_cli_version() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert __version__ in res.stdout


def test_cli_model_list() -> None:
    res = runner.invoke(app, ["model", "list"])
    assert res.exit_code == 0
    assert "spafm-s-v0" in res.stdout


def test_cli_model_info() -> None:
    res = runner.invoke(app, ["model", "info", "spafm-s-v0"])
    assert res.exit_code == 0
    assert "spafm-s-v0" in res.stdout
    assert "MIT" in res.stdout


def test_cli_model_info_unknown() -> None:
    res = runner.invoke(app, ["model", "info", "nope"])
    assert res.exit_code != 0
