# SPDX-License-Identifier: Apache-2.0
"""The sharded_load local fallback must engage only on the weight-index gate
rejecting a local checkpoint, and leave every other outcome untouched."""

import pytest

from omlx.patches import mlx_lm_sharded_load as patch_module
from omlx.patches.mlx_lm_sharded_load import (
    _wrap,
    install_local_sharded_load_fallback,
)

GATE_ERROR = ValueError("Pipeline loading is only supported for MLX converted models.")


def test_a_successful_original_passes_through():
    wrapped = _wrap(lambda repo, *a, **k: ("model", "tokenizer"))
    assert wrapped("/anywhere") == ("model", "tokenizer")


def test_the_gate_error_on_a_local_directory_falls_back(tmp_path, monkeypatch):
    def original(repo, *a, **k):
        raise GATE_ERROR

    sentinel = object()
    monkeypatch.setattr(
        patch_module, "_local_sharded_load", lambda repo, *a, **k: sentinel
    )
    assert _wrap(original)(str(tmp_path)) is sentinel


def test_the_gate_error_on_a_remote_repo_still_raises(monkeypatch):
    def original(repo, *a, **k):
        raise GATE_ERROR

    monkeypatch.setattr(
        patch_module, "_local_sharded_load", lambda *a, **k: pytest.fail("fell back")
    )
    with pytest.raises(ValueError, match="MLX converted"):
        _wrap(original)("mlx-community/some-remote-model")


def test_other_errors_are_not_swallowed(tmp_path, monkeypatch):
    def original(repo, *a, **k):
        raise ValueError("does not support any sharding")

    monkeypatch.setattr(
        patch_module, "_local_sharded_load", lambda *a, **k: pytest.fail("fell back")
    )
    with pytest.raises(ValueError, match="sharding"):
        _wrap(original)(str(tmp_path))


def test_install_wraps_both_module_bindings_once(monkeypatch):
    import mlx_lm.server as server_module
    import mlx_lm.utils as utils_module

    def original(repo, *a, **k):
        return "loaded"

    monkeypatch.setattr(utils_module, "sharded_load", original)
    monkeypatch.setattr(server_module, "sharded_load", original)

    assert install_local_sharded_load_fallback() is True
    assert utils_module.sharded_load is server_module.sharded_load
    assert getattr(utils_module.sharded_load, "_omlx_local_fallback", False)
    assert utils_module.sharded_load("/x") == "loaded"
    # A second install is a no-op, not a double wrap.
    assert install_local_sharded_load_fallback() is False
