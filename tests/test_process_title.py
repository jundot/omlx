# SPDX-License-Identifier: Apache-2.0
"""Tests for process title helpers."""

import builtins
import sys
from types import ModuleType
from unittest.mock import MagicMock

from omlx.process_title import DEFAULT_PROCESS_TITLE, set_process_title


def test_default_process_title_is_server_name():
    assert DEFAULT_PROCESS_TITLE == "omlx-server"


def test_set_process_title_fallback_updates_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["python", "-m", "omlx.cli", "serve"])

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "setproctitle":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert set_process_title() is False
    assert sys.argv[0] == "omlx-server"


def test_set_process_title_uses_native_module(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["python", "-m", "omlx.cli", "serve"])

    fake_module = ModuleType("setproctitle")
    fake_module.setproctitle = MagicMock()
    monkeypatch.setitem(sys.modules, "setproctitle", fake_module)

    assert set_process_title() is True
    assert sys.argv[0] == "omlx-server"
    fake_module.setproctitle.assert_called_once_with("omlx-server")
