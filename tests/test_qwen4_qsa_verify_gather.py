# SPDX-License-Identifier: Apache-2.0
"""Lightning MTP verify rows through the gathered QSA path (batch-one text)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from tests.test_qwen4_qsa_decode_gather import _tiny_text_config


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _layer_and_prefix(seed: int = 19, prefix_tokens: int = 10):
    import mlx_vlm.models.qwen4_exp.language as language

    config = _tiny_text_config()
    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    fast_cache = language.QSAKVCache()
    reference_cache = language.QSAKVCache()
    mx.random.seed(seed)
    prefix = mx.random.normal((1, prefix_tokens, config.hidden_size))
    mx.eval(attention(prefix, mask="causal", cache=fast_cache), attention(prefix, mask="causal", cache=reference_cache))
    return config, attention, fast_cache, reference_cache


@pytest.mark.parametrize("rows", [2, 4])
def test_qwen4_verify_rows_gather_selected_blocks_and_match_official(monkeypatch, rows):
    import mlx_vlm.models.qwen4_exp.language as language

    config, attention, fast_cache, reference_cache = _layer_and_prefix()
    verify = mx.random.normal((1, rows, config.hidden_size))

    gathered_query_tokens = []
    original = language.contiguous_causal_gathered_qsa  # bound at import; patch the name language.py calls

    def tracked(queries, *args, **kwargs):
        gathered_query_tokens.append(int(queries.shape[2]))
        return original(queries, *args, **kwargs)

    monkeypatch.setattr(language, "contiguous_causal_gathered_qsa", tracked)
    actual = attention(verify, mask="causal", cache=fast_cache, target_verify=True)

    # Reference: the official masked-dense verify path.
    monkeypatch.setattr(
        language.Qwen4ExpAttention, "_gathered_text_verify_eligible", lambda *a, **k: False, raising=False
    )
    expected = attention(verify, mask="causal", cache=reference_cache, target_verify=True)
    mx.eval(actual, expected)

    # key_len=10+rows > budget 8, so the verify rows must attend gathered blocks, once, for all rows.
    assert gathered_query_tokens == [rows]
    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()
    assert mx.array_equal(mx.argmax(actual, axis=-1), mx.argmax(expected, axis=-1)).item()
    assert fast_cache.offset == reference_cache.offset == 10 + rows
    for fast_value, reference_value in zip(fast_cache.state, reference_cache.state):
        assert mx.array_equal(fast_value, reference_value).item()


def test_qwen4_verify_rows_survive_rollback_like_official(monkeypatch):
    """After a rejected draft the caches are trimmed; the next verify must still match the official path."""
    import mlx_vlm.models.qwen4_exp.language as language

    config, attention, fast_cache, reference_cache = _layer_and_prefix(seed=23)
    first = mx.random.normal((1, 4, config.hidden_size))
    mx.eval(attention(first, mask="causal", cache=fast_cache, target_verify=True))
    monkeypatch.setattr(
        language.Qwen4ExpAttention, "_gathered_text_verify_eligible", lambda *a, **k: False, raising=False
    )
    mx.eval(attention(first, mask="causal", cache=reference_cache, target_verify=True))
    monkeypatch.undo()
    for cache in (fast_cache, reference_cache):
        cache.trim(3)  # only the first of the four rows was accepted
    assert fast_cache.offset == reference_cache.offset == 11

    second = mx.random.normal((1, 4, config.hidden_size))
    actual = attention(second, mask="causal", cache=fast_cache, target_verify=True)
    monkeypatch.setattr(
        language.Qwen4ExpAttention, "_gathered_text_verify_eligible", lambda *a, **k: False, raising=False
    )
    expected = attention(second, mask="causal", cache=reference_cache, target_verify=True)
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()
    for fast_value, reference_value in zip(fast_cache.state, reference_cache.state):
        assert mx.array_equal(fast_value, reference_value).item()


def test_qwen4_verify_gather_kill_switch_keeps_official_path(monkeypatch):
    import mlx_vlm.models.qwen4_exp.language as language

    config, attention, fast_cache, _ = _layer_and_prefix()
    calls = []
    original = language.contiguous_causal_gathered_qsa
    monkeypatch.setattr(language, "contiguous_causal_gathered_qsa", lambda *a, **k: calls.append(1) or original(*a, **k))
    monkeypatch.setattr(language, "_GATHERED_VERIFY_DISABLED", True)
    mx.eval(attention(mx.random.normal((1, 4, config.hidden_size)), mask="causal", cache=fast_cache, target_verify=True))
    assert calls == []


def test_qwen4_verify_gather_requires_rank_two_positions():
    """The adapter emits rank-two text positions only above the step threshold; the
    verify arm must not engage on rank-three broadcast planes below it (measured
    -14% adaptive tok/s at 4k when it did)."""
    import mlx_vlm.models.qwen4_exp.language as language

    config, attention, fast_cache, _ = _layer_and_prefix()
    rows = 4
    verify = mx.random.normal((1, rows, config.hidden_size))
    text = mx.arange(fast_cache.offset, fast_cache.offset + rows)[None, :]
    planes = mx.broadcast_to(text[None, :, :], (3, 1, rows))
    eligible = lambda positions: attention._gathered_text_verify_eligible(
        verify, "causal", fast_cache, positions, None, True
    )
    assert fast_cache.offset + rows > attention.indexer.token_budget
    assert eligible(text) is True
    assert eligible(None) is True
    assert eligible(planes) is False
