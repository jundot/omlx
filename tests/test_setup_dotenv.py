# SPDX-License-Identifier: Apache-2.0
"""Cover setup.py's .env loader — the local opt-in for custom kernel builds."""

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_setup_module():
    """Import setup.py by path; importing it does not run setup()."""
    spec = importlib.util.spec_from_file_location(
        "omlx_setup_under_test", REPO_ROOT / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup_module():
    return _load_setup_module()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OMLX_WITH_CUSTOM_KERNEL", raising=False)


def test_loads_flag_from_dotenv(setup_module, tmp_path):
    env = tmp_path / ".env"
    env.write_text("OMLX_WITH_CUSTOM_KERNEL=1\n")

    setup_module._load_dotenv(env)

    assert os.environ["OMLX_WITH_CUSTOM_KERNEL"] == "1"
    assert setup_module._with_custom_kernel() is True


def test_real_env_wins_over_dotenv(setup_module, tmp_path, monkeypatch):
    """CI and one-off `OMLX_WITH_CUSTOM_KERNEL=0 pip install` must override."""
    monkeypatch.setenv("OMLX_WITH_CUSTOM_KERNEL", "0")
    env = tmp_path / ".env"
    env.write_text("OMLX_WITH_CUSTOM_KERNEL=1\n")

    setup_module._load_dotenv(env)

    assert os.environ["OMLX_WITH_CUSTOM_KERNEL"] == "0"
    assert setup_module._with_custom_kernel() is False


def test_skips_comments_blanks_quotes_and_export(setup_module, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "\n"
        "# a comment\n"
        "  \n"
        "not_a_pair\n"
        'export OMLX_WITH_CUSTOM_KERNEL="1"\n'
        "OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET = '15.0'\n"
    )

    setup_module._load_dotenv(env)

    assert os.environ["OMLX_WITH_CUSTOM_KERNEL"] == "1"
    assert os.environ["OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET"] == "15.0"


def test_missing_dotenv_is_not_an_error(setup_module, tmp_path):
    setup_module._load_dotenv(tmp_path / "nope.env")

    assert "OMLX_WITH_CUSTOM_KERNEL" not in os.environ
