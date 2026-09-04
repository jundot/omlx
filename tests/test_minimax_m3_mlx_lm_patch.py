# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 must be loadable by mlx-lm, or the cluster cannot serve it.

Every cluster rank is an ``mlx_lm.server``. Pinned mlx-lm has no
``minimax_m3_vl``, so a 225 GiB model that fits two Macs with room to spare was
unservable across them while fitting one Mac only at ~1k tokens of context.
"""

from __future__ import annotations

import pytest

# The vendored MiniMax implementation needs mlx-vlm; a runner without it
# should skip these, not error collecting them.
pytest.importorskip("mlx_vlm")

from omlx.patches.minimax_m3_mlx_lm import (
    apply_minimax_m3_mlx_lm_patch,
    is_minimax_m3,
)

# Shaped from the real mlx-community/MiniMax-M3-4bit config: the language
# dimensions live under text_config, which is the thing that must be unwrapped.
CONFIG = {
    "model_type": "minimax_m3_vl",
    "text_config": {
        "num_hidden_layers": 4,
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 3072,
        "shared_intermediate_size": 3072,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "vocab_size": 1024,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000,
        "max_position_embeddings": 1048576,
    },
}


def test_the_patch_reports_which_models_it_is_for():
    assert is_minimax_m3({"model_type": "minimax_m3_vl"})
    assert is_minimax_m3({"model_type": "minimax_m3"})
    assert not is_minimax_m3({"model_type": "qwen3_5"})
    assert not is_minimax_m3({})


def test_mlx_lm_cannot_load_minimax_without_the_patch():
    """States the gap the patch closes, so its removal is noticed."""

    import sys

    if "mlx_lm.models.minimax_m3_vl" in sys.modules:
        pytest.skip("patch already applied in this process")

    from mlx_lm.utils import _get_classes

    with pytest.raises(ValueError, match="not supported"):
        _get_classes({"model_type": "minimax_m3_vl"})


def test_mlx_lm_resolves_minimax_after_the_patch():
    from mlx_lm.utils import _get_classes

    assert apply_minimax_m3_mlx_lm_patch()
    model_cls, args_cls = _get_classes({"model_type": "minimax_m3_vl"})
    assert model_cls.__name__ == "Model"
    assert hasattr(args_cls, "from_dict")


def test_applying_twice_is_harmless():
    assert apply_minimax_m3_mlx_lm_patch()
    assert apply_minimax_m3_mlx_lm_patch()


def test_the_language_dimensions_are_read_from_the_nested_config():
    from mlx_lm.utils import _get_classes

    apply_minimax_m3_mlx_lm_patch()
    _, args_cls = _get_classes(CONFIG)
    args = args_cls.from_dict(CONFIG)
    assert args.num_hidden_layers == 4
    assert args.num_key_value_heads == 4
    assert args.head_dim == 128


def test_a_pipeline_stage_reaches_the_tree_that_actually_runs():
    """The critical one: a rank must not report a stage while running it all.

    The original version of this test asserted the *buggy* contract — that
    ``model.layers`` is an assignable list. Storing that list gave the wrapper
    its own dict entry pointing at the complete model, so every rank loaded
    all ~225 GiB (audit finding 1). ``layers`` is now a read-only view of the
    tree that executes: it must reflect a ``pipeline()`` slice instantly, and
    an assignment — which could only ever detach the two — must raise.
    """

    import pytest
    from mlx_lm.utils import _get_classes

    apply_minimax_m3_mlx_lm_patch()
    model_cls, args_cls = _get_classes(CONFIG)
    model = model_cls(args_cls.from_dict(CONFIG))

    assert len(model.layers) == 4

    class _Group:
        def rank(self) -> int:
            return 0

        def size(self) -> int:
            return 2

    model.model.pipeline(_Group())
    assert model.layers is model.inner.language_model.model.layers, (
        "the wrapper must expose the very list that executes, not a copy"
    )
    assert sum(1 for layer in model.layers if layer is not None) == 2

    with pytest.raises(AttributeError):
        model.layers = []


def test_adapter_exposes_an_explicit_rank_zero_logits_contract():
    """Worker ranks may skip MiniMax's large vocabulary projection safely."""

    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_lm.utils import _get_classes

    apply_minimax_m3_mlx_lm_patch()
    model_cls, _ = _get_classes(CONFIG)

    calls = []

    class Inner:
        def __call__(self, inputs, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(logits=None)

    adapter = SimpleNamespace(inner=Inner())
    result = model_cls.__call__(
        adapter,
        mx.array([[1]], dtype=mx.uint32),
        cache=["cache"],
        skip_logits=True,
    )

    assert model_cls._omlx_supports_rank_zero_logits is True
    assert result is None
    assert calls == [
        {
            "mask": None,
            "cache": ["cache"],
            "skip_logits": True,
        }
    ]


def test_a_failed_registration_leaves_no_broken_module_behind(monkeypatch):
    """A husk in sys.modules turns a missing dep into a confusing AttributeError.

    Seen on a peer whose venv lacked mlx_vlm: registration failed, but the
    empty module stayed registered, so mlx-lm found ``minimax_m3_vl`` with no
    ``Model`` and failed far from the real cause.
    """

    import sys

    from omlx.patches import minimax_m3_mlx_lm as patch

    sys.modules.pop(patch._QUALNAME, None)

    def _boom(module):
        raise ModuleNotFoundError("No module named 'mlx_vlm'")

    importlib_spec = __import__("importlib.util", fromlist=["util"])
    monkeypatch.setattr(
        importlib_spec, "module_from_spec",
        lambda spec: type(sys)(patch._QUALNAME),
    )
    monkeypatch.setattr(patch, "_register_module", patch._register_module)

    class _Loader:
        def exec_module(self, module):
            _boom(module)

    class _Spec:
        loader = _Loader()

    monkeypatch.setattr(
        importlib_spec, "spec_from_file_location", lambda *a, **k: _Spec()
    )

    assert patch.apply_minimax_m3_mlx_lm_patch() is False
    assert patch._QUALNAME not in sys.modules, "the husk must be removed"
