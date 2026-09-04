from __future__ import annotations

from types import SimpleNamespace

from omlx.patches.mlx_lm_mtp.batch_generator import _trunk_norm_module


def _norm():
    return lambda x: ("normed", x)


class TestTrunkNormModule:
    def test_unmarked_model_resolves_inner_norm(self):
        norm = _norm()
        model = SimpleNamespace(model=SimpleNamespace(norm=norm))
        assert _trunk_norm_module(model) is norm

    def test_unmarked_language_model_wrapper(self):
        norm = _norm()
        inner = SimpleNamespace(model=SimpleNamespace(norm=norm))
        model = SimpleNamespace(language_model=inner)
        assert _trunk_norm_module(model) is norm

    def test_marked_instance_returns_identity(self):
        model = SimpleNamespace(
            _omlx_mtp_head_hidden_normed=True,
            model=SimpleNamespace(norm=_norm()),
        )
        fn = _trunk_norm_module(model)
        sentinel = object()
        assert fn(sentinel) is sentinel

    def test_marked_inner_language_model_returns_identity(self):
        inner = SimpleNamespace(
            _omlx_mtp_head_hidden_normed=True,
            model=SimpleNamespace(norm=_norm()),
        )
        model = SimpleNamespace(language_model=inner)
        fn = _trunk_norm_module(model)
        sentinel = object()
        assert fn(sentinel) is sentinel
