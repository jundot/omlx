# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the Cluster Assistant API tests.

The ``api`` fixture is defined here rather than in a test module so both
``test_apply_api.py`` and ``test_start_api.py`` get it without importing one
test module from another (which also keeps the fixture visible to lint).
"""

from __future__ import annotations

import contextlib

import pytest
from assistant_harness import assistant_api


@pytest.fixture()
def api(tmp_path, monkeypatch):
    """Return a builder for a router whose Mac-facing seams are all fakes."""

    stack = contextlib.ExitStack()

    def _build(**kwargs):
        return stack.enter_context(assistant_api(tmp_path, monkeypatch, **kwargs))

    yield _build
    stack.close()
