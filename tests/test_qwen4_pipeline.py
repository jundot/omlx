# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp rank stages: local layers, local masks, one HC handoff."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


class _Group:
    """The two attributes PipelineMixin reads, without a real ring."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _language_model():
    from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config

    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    return LanguageModel(config.text_config, config)


def _stage(monkeypatch, *, rank: int, size: int):
    sent: list[tuple[int, tuple[int, ...]]] = []

    monkeypatch.setattr(mx.distributed, "recv_like", lambda value, src, **k: value)
    monkeypatch.setattr(
        mx.distributed,
        "send",
        lambda value, dst, **k: sent.append((dst, tuple(value.shape))) or value,
    )
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **k: value)

    model = _language_model()
    model.model.pipeline(_Group(rank, size))
    return model, sent


def _stage_cache(model):
    from mlx_vlm.models.qwen4_exp.language import ArraysCache, QSAKVCache

    return [
        (
            ArraysCache(size=4 if "ple" in layer else 2)
            if layer.is_linear
            else QSAKVCache()
        )
        for layer in model.pipeline_layers
    ]


def test_a_head_stage_with_only_linear_layers_needs_no_full_attention_mask(
    monkeypatch,
):
    model, sent = _stage(monkeypatch, rank=1, size=2)

    assert model.model.fa_idx is None
    assert model.model.ssm_idx == 0
    assert len(model.make_cache()) == 1

    out = model.model(
        mx.array([[2, 3, 4]], dtype=mx.int32),
        cache=model.make_cache(),
    )

    hidden = model.args.hidden_size
    assert out.shape == (1, 3, hidden)
    assert sent[0][0] == 0
    assert sent[0][1][-1] == model.args.hc_count * hidden


def test_a_tail_stage_with_only_full_attention_layers_needs_no_ssm_mask(
    monkeypatch,
):
    model, sent = _stage(monkeypatch, rank=0, size=2)

    assert model.model.ssm_idx is None
    assert model.model.fa_idx == 0
    assert len(model.make_cache()) == 1

    out = model.model(
        mx.array([[2, 3, 4]], dtype=mx.int32),
        cache=model.make_cache(),
    )

    assert out.shape == (1, 3, model.args.hidden_size)
    assert sent == []


def test_single_stage_keeps_the_ordinary_forward(monkeypatch):
    model, sent = _stage(monkeypatch, rank=0, size=1)

    assert model.model.pipeline_layers == model.model.layers
    assert model.model.fa_idx is not None
    assert model.model.ssm_idx is not None

    out = model.model(
        mx.array([[2, 3, 4]], dtype=mx.int32),
        cache=model.make_cache(),
    )

    assert out.shape == (1, 3, model.args.hidden_size)
    assert sent == []


def test_two_stage_hc_seam_matches_the_full_forward(monkeypatch):
    """The handoff must reproduce the single-model mixer, not just its shape."""

    import copy

    from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config

    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    mx.random.seed(7)
    full = LanguageModel(config.text_config, config).model
    head = copy.deepcopy(full)
    tail = copy.deepcopy(full)

    head.pipeline(_Group(1, 2))
    tail.pipeline(_Group(0, 2))

    transferred = []

    def fake_send(value, dst, **kwargs):
        mx.eval(value)
        transferred.append(value)
        return value

    monkeypatch.setattr(mx.distributed, "send", fake_send)
    monkeypatch.setattr(
        mx.distributed, "recv_like", lambda value, src, **k: transferred.pop(0)
    )
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **k: value)

    inputs = mx.array([[2, 3, 4]], dtype=mx.int32)
    head_out = head(inputs, cache=_stage_cache(head))
    tail_out = tail(inputs, cache=_stage_cache(tail))
    full_out = full(inputs, cache=_stage_cache(full))

    mx.eval(head_out, tail_out, full_out)

    assert tail_out.shape == full_out.shape
    assert mx.allclose(tail_out, full_out, atol=2e-2).item()
