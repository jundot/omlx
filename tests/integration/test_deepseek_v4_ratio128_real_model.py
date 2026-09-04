# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint coverage for DeepSeek V4 ratio-128 attention.

These tests never download models. They load explicitly selected local
checkpoints through oMLX's public text-model loader and execute a 257-token
prefill plus a 17-token cached continuation so every ratio-128 layer exercises
pooled KV masks at both zero and nonzero offsets.

Run each checkpoint in its own process to keep the memory boundary explicit:

    OMLX_DEEPSEEK_V4_HIGH_BIT_MODEL_PATH=/path/to/DeepSeek-V4-Flash-0731 \
        uv run pytest tests/integration/test_deepseek_v4_ratio128_real_model.py \
        -m slow -k high-bit -s -q

    OMLX_DEEPSEEK_V4_SUB4_MODEL_PATH=/path/to/DeepSeek-V4-Flash-0731-oQ2.5e \
        uv run pytest tests/integration/test_deepseek_v4_ratio128_real_model.py \
        -m slow -k sub-four-bit -s -q
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="DeepSeek V4 MLX integration requires macOS on Apple Silicon.",
    ),
]

_PREFILL_TOKENS = 257
_CONTINUATION_TOKENS = 17


def _configured_checkpoint(environment_variable: str, *, expect_sub4: bool) -> Path:
    configured = os.environ.get(environment_variable)
    if not configured:
        pytest.skip(f"Set {environment_variable} to run this real-model test.")

    model_path = Path(configured).expanduser()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        pytest.fail(f"{environment_variable} has no config.json: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type.startswith("deepseek_v4"):
        pytest.fail(
            f"{environment_variable} must identify a DeepSeek V4 checkpoint, "
            f"not model_type={model_type!r}: {model_path}"
        )

    quantizations = [config.get("quantization"), config.get("quantization_config")]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        quantizations.append(text_config.get("quantization_config"))
    declared_bits = [
        quantization.get("bits")
        for quantization in quantizations
        if isinstance(quantization, dict)
        and isinstance(quantization.get("bits"), (int, float))
        and not isinstance(quantization.get("bits"), bool)
    ]
    has_sub4 = any(float(bits) < 4 for bits in declared_bits)
    assert has_sub4 is expect_sub4, (
        f"{environment_variable} quantization does not match the requested lane: "
        f"declared bits={declared_bits!r}, expect_sub4={expect_sub4}."
    )
    return model_path


@pytest.mark.parametrize(
    ("environment_variable", "expect_native"),
    (
        ("OMLX_DEEPSEEK_V4_HIGH_BIT_MODEL_PATH", True),
        ("OMLX_DEEPSEEK_V4_SUB4_MODEL_PATH", False),
    ),
    ids=("high-bit-native", "sub-four-bit-reference"),
)
def test_real_checkpoint_prefill_selects_ratio128_attention_policy(
    monkeypatch, environment_variable, expect_native
):
    import mlx.core as mx

    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.utils.model_loading import load_text_model

    model_path = _configured_checkpoint(
        environment_variable,
        expect_sub4=not expect_native,
    )
    if expect_native:
        assert fast.has_symbol("deepseek_v4_sparse_attention"), (
            "The high-bit integration lane requires the compiled "
            "deepseek_v4_sparse_attention kernel."
        )

    model = tokenizer = cache = logits = last_logits = None
    continuation_logits = continuation_last_logits = None
    try:
        model, tokenizer = load_text_model(str(model_path))
        assert model.args.use_native_ratio128_attention is expect_native

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        ratio128_helper_calls = 0
        ratio128_native_calls = 0
        original_sparse = dsv4._sparse_pooled_attention
        original_native = fast.deepseek_v4_sparse_attention

        def sparse_spy(*args, **kwargs):
            nonlocal ratio128_helper_calls
            if kwargs.get("compress_ratio") == 128:
                ratio128_helper_calls += 1
            return original_sparse(*args, **kwargs)

        def native_spy(
            q,
            local_kv,
            pooled,
            topk_indices,
            sinks,
            scale,
            q_offset,
            compress_ratio,
            local_window,
            *,
            stream=None,
        ):
            nonlocal ratio128_native_calls
            if compress_ratio == 128:
                ratio128_native_calls += 1
            return original_native(
                q,
                local_kv,
                pooled,
                topk_indices,
                sinks,
                scale,
                q_offset,
                compress_ratio,
                local_window,
                stream=stream,
            )

        monkeypatch.setattr(dsv4, "_sparse_pooled_attention", sparse_spy)
        monkeypatch.setattr(fast, "deepseek_v4_sparse_attention", native_spy)
        monkeypatch.setattr(
            dsv4,
            "_DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED",
            False,
        )

        cache = model.make_cache()
        input_ids = mx.arange(_PREFILL_TOKENS, dtype=mx.int32)[None]
        logits = model(input_ids, cache=cache)
        last_logits = logits[:, -1]
        continuation_ids = mx.arange(
            _PREFILL_TOKENS,
            _PREFILL_TOKENS + _CONTINUATION_TOKENS,
            dtype=mx.int32,
        )[None]
        continuation_logits = model(continuation_ids, cache=cache)
        continuation_last_logits = continuation_logits[:, -1]
        mx.eval(last_logits, continuation_last_logits)

        ratio128_layers = sum(ratio == 128 for ratio in model.args.compress_ratios)
        assert last_logits.shape == (1, model.args.vocab_size)
        assert continuation_last_logits.shape == (1, model.args.vocab_size)
        assert mx.all(mx.isfinite(last_logits)).item()
        assert mx.all(mx.isfinite(continuation_last_logits)).item()
        if expect_native:
            assert ratio128_helper_calls == ratio128_layers * 2
            assert ratio128_native_calls == ratio128_layers * 2
            assert dsv4._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED is False
        else:
            assert ratio128_helper_calls == 0
            assert ratio128_native_calls == 0
    finally:
        continuation_last_logits = continuation_logits = None
        last_logits = logits = cache = tokenizer = model = None
        gc.collect()
        mx.clear_cache()
