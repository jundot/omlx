import importlib
import sys
from types import SimpleNamespace

import mlx.core as mx


class _FakeArray:
    def __init__(self, shape, dtype=mx.bfloat16):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype


class TestQwen35RaggedSDPAGuard:
    def _fresh_guard(self):
        import omlx.patches.qwen35_ragged_sdpa_guard as guard

        return importlib.reload(guard)

    def _install_fake_language(self, monkeypatch):
        calls = []

        def original(queries, keys, values, pads, scale):
            calls.append((queries, keys, values, pads, scale))
            return "fast-output"

        fake_lang = SimpleNamespace(
            _qwen3_5_ragged_decode_attention=original,
            _qwen3_5_sdpa_vector_plan=lambda seq_len, q_heads, kv_heads: (
                "two_pass",
                128,
            ),
        )
        monkeypatch.setitem(
            sys.modules, "mlx_vlm.models.qwen3_5.language", fake_lang
        )
        return fake_lang, calls

    def test_unsupported_variant_falls_back_without_launching_fast_path(self, monkeypatch):
        guard = self._fresh_guard()
        fake_lang, calls = self._install_fake_language(monkeypatch)
        monkeypatch.setattr(guard.mx.metal, "is_available", lambda: True)
        monkeypatch.setattr(guard, "_variant_supported", lambda *args: False)

        assert guard.apply_qwen35_ragged_sdpa_guard_patch() is True

        out = fake_lang._qwen3_5_ragged_decode_attention(
            _FakeArray((2, 8, 1, 256)),
            _FakeArray((2, 4, 1500, 256)),
            _FakeArray((2, 4, 1500, 256)),
            [0, 16],
            1.0,
        )

        assert out is None
        assert calls == []

    def test_supported_variant_uses_original_fast_path(self, monkeypatch):
        guard = self._fresh_guard()
        fake_lang, calls = self._install_fake_language(monkeypatch)
        monkeypatch.setattr(guard.mx.metal, "is_available", lambda: True)
        seen = []

        def supported(*args):
            seen.append(args)
            return True

        monkeypatch.setattr(guard, "_variant_supported", supported)

        assert guard.apply_qwen35_ragged_sdpa_guard_patch() is True

        out = fake_lang._qwen3_5_ragged_decode_attention(
            _FakeArray((2, 8, 1, 256)),
            _FakeArray((2, 4, 1500, 256)),
            _FakeArray((2, 4, 1500, 256)),
            [0, 16],
            1.0,
        )

        assert out == "fast-output"
        assert len(calls) == 1
        assert seen == [(mx.bfloat16, 256, 256, 8, 4, 2, 1500, "two_pass", 128)]

    def test_detects_threadgroup_limit_error(self):
        guard = self._fresh_guard()
        exc = ValueError(
            "Thread group size (1024) is greater than the maximum allowed "
            "threads per threadgroup (896)."
        )

        assert guard._is_threadgroup_limit_error(exc) is True
