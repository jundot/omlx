# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import mlx.core as mx

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

apply_deepseek_v4_patch()

import mlx_lm.models.deepseek_v4 as dm  # noqa: E402
import mlx_lm.models.hyper_connection as hc  # noqa: E402


class _Cache:
    offset = 0
    _offset = 0
    state = ()


def test_canonical_attention_splits_only_the_exact_2k_contract(monkeypatch):
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_CANONICAL_WIDE_PREFILL", True)
    dm._DEEPSEEK_V4_CANONICAL_WIDE_PREFILL_STATE.active = False
    module = SimpleNamespace(
        training=False,
        compress_ratio=4,
        config=SimpleNamespace(sliding_window=128),
    )
    value = mx.broadcast_to(
        mx.zeros((1, 1, 4096), dtype=mx.bfloat16),
        (1, 2048, 4096),
    )
    calls = []

    def call(part, mask, cache, *, _standard_mask):
        calls.append((tuple(part.shape), mask is not None, _standard_mask))
        return part

    output = dm._canonical_wide_attention_prefill(
        module,
        value,
        _Cache(),
        standard_mask=True,
        call=call,
    )

    assert output is not None
    assert tuple(output.shape) == (1, 2048, 4096)
    assert calls == [
        ((1, 1024, 4096), True, True),
        ((1, 1024, 4096), True, True),
    ]
    module.compress_ratio = 0
    assert (
        dm._canonical_wide_attention_prefill(
            module,
            value,
            _Cache(),
            standard_mask=True,
            call=call,
        )
        is None
    )


def test_canonical_hyperconnection_preserves_two_1k_calls(monkeypatch):
    monkeypatch.setattr(hc, "_CANONICAL_WIDE_PREFILL", True)
    value = mx.broadcast_to(
        mx.zeros((1, 1, 4, 4096), dtype=mx.bfloat16),
        (1, 2048, 4, 4096),
    )
    calls = []

    class FakeHyperConnection:
        training = False
        hc_mult = 4

        def _call_one(self, part):
            calls.append(tuple(part.shape))
            row = part[:, :, 0, :1]
            return (
                row,
                mx.zeros((1, part.shape[1], 4)),
                mx.zeros((1, part.shape[1], 4, 4)),
            )

    outputs = hc.HyperConnection.__call__(FakeHyperConnection(), value)

    assert calls == [(1, 1024, 4, 4096), (1, 1024, 4, 4096)]
    assert [tuple(output.shape) for output in outputs] == [
        (1, 2048, 1),
        (1, 2048, 4),
        (1, 2048, 4, 4),
    ]


def test_compiled_hc_decode_producer_is_exact_shape_only(monkeypatch):
    monkeypatch.setattr(hc, "_COMPILED_HC_DECODE_PRODUCER", True)
    module = SimpleNamespace(
        training=False,
        fn=mx.zeros((24, 16384), dtype=mx.float32),
        norm_eps=1e-6,
    )
    decode = mx.zeros((1, 1, 4, 4096), dtype=mx.bfloat16)

    assert hc._can_use_compiled_hc_decode_producer(module, decode)
    module.training = True
    assert not hc._can_use_compiled_hc_decode_producer(module, decode)
    module.training = False
    assert not hc._can_use_compiled_hc_decode_producer(
        module,
        mx.zeros((1, 2, 4, 4096), dtype=mx.bfloat16),
    )
